use std::sync::Arc;
use std::sync::mpsc::{Receiver};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::Version;
use winit::window::{Window, WindowBuilder};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::windows::{WindowBuilderExtWindows};
use std::f32::consts::PI;


pub fn window_loop(receiver: Receiver<Vec<f32>>) {

    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name.as_ref().unwrap(),
        physical.properties().device_type.unwrap(),
    );
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_drag_and_drop(false) // otherwise conflicts with WASAPI
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec3 position;
                layout(location = 0) out vec4 out_position;

				void main() {
                    out_position = vec4(position, 1.0); 
					gl_Position = vec4(position, 1.0);
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450

                layout(location = 0) in vec4 position;
				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(
                        min(position.z, 1),
                        min(position.z*3 - 1, 1.0),
                        min(position.z*3 - 2, 1.0),
                        1.0);
				}
			"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .line_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    );
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let fft: Vec<f32>;
                match receiver.recv() {
                    Ok(chunk) => {
                        fft = chunk;
                    }
                    Err(err) => return,
                }
                let log_base = 1.015f32;
                print!("{}", log_base);
                
                let vertex_buffer = {
                    #[derive(Default, Debug, Clone)]
                    struct Vertex {
                        position: [f32; 3],
                    }
                    vulkano::impl_vertex!(Vertex, position);
                    
                    CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::all(),
                        false,
                        fft
                        .iter()
                        .enumerate()
                        //.map(|(i, x)| Vertex { position: [log_base.powi(i as i32) * 2f32 - 1f32, x/255f32] })
                        //.map(|(i, x)| Vertex { position: [i as f32 / fft.len() as f32 * 2f32 - 1f32, - x/255f32 + 0.5f32] })
                        //.map(|(i, x)| Vertex { position: [(i as f32).log(1.1f32) / (fft.len() as f32).log(1.1f32) * 2f32 - 1f32, - x/255f32 + 0.5f32, *x/255f32] })
                        .map(|(i, x)| Vertex { position: gen_vertex_quasar(fft.len(), i, *x) })
                        .collect::<Vec<Vertex>>()
                        .iter()
                        .cloned(),
                    )
                    .unwrap()
                };            

                let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        (),
                        (),
                        vec![],
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}


fn gen_vertex_flatvis(len: usize, i: usize, value: f32) -> [f32; 3] {
    [(i as f32).log(1.1f32) / (len as f32).log(1.1f32) * 2f32 - 1f32,
     - value/255f32 + 0.5f32,
     value/255f32]
}


fn gen_vertex_quasar(len: usize, i: usize, value: f32) -> [f32; 3] {
    let len = len as f32;
    let i = i as f32;
    //let (x, y) = ((i as f32).sin(), (i as f32).cos());
    let circle_progress = i; //i.log2() * PI * 256f32;
    let (x, y) = (i.sin(), i.cos());
    [
        x * i/len,
        y * i/len,
        (value/255f32).min(1f32).sqrt()
    ]
}


fn gen_vertex_radial(len: usize, i: usize, value: f32) -> [f32; 3] {
    let circle_progress = (i as f32).log2() * PI*2f32;
    let (x, y) = (circle_progress.sin(), circle_progress.cos());
    let log_factor = (i as f32).log(1.5f32) / (len as f32).log(1.5f32);
    [
        x * log_factor - x * value / 10000f32,
        y * log_factor - y * value / 10000f32,
        (value/255f32).min(1f32).sqrt()
    ]
}


fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
