use std::sync::Arc;
use std::sync::mpsc::{Receiver};
use std::time::Duration;


use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, Features
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};

use winit::window::{Window, WindowBuilder, Fullscreen};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent, VirtualKeyCode, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
#[cfg(target_os = "windows")] use winit::platform::windows::{WindowBuilderExtWindows};
use std::f32::consts::PI;
use cgmath::{ Vector3, Vector4, Point3, Matrix4, Rad };
use std::convert::TryFrom;
use std::time::Instant;


pub fn window_loop(receiver: Receiver<Vec<f32>>) {
    
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();
    
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let mut builder = WindowBuilder::new();
    #[cfg(target_os = "windows")]{
        builder = builder.with_drag_and_drop(false); // otherwise conflicts with WASAPI
    }
    let event_loop = EventLoop::new();
    let surface = builder
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let enabled_features = Features {
        wide_lines: true,
        .. Features::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // Some devices may not support the extensions or features that your application, or
            // report properties and limits that are not sufficient for your application. These
            // should be filtered out here.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will execute
            // our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a draw
            // queue and a compute queue), similar to CPU threads. This is something you have to
            // have to manage manually in Vulkan. Queues of the same type belong to the same
            // queue family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-life application, you may want to use a separate dedicated transfer queue to
            // handle data transfers in parallel with graphics operations. You may also need a
            // separate queue for compute operations, if your application uses those.
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // We select a queue family that supports graphics operations. When drawing to
                    // a window surface, as we do in this example, we also need to check that queues
                    // in this queue family are capable of presenting images to the surface.
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
                    && p.supported_features().contains(&enabled_features)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|i| (p, i as u32))
        })
        // All the physical devices that pass the filters above are suitable for the application.
        // However, not every device is equal, some are preferred over others. Now, we assign
        // each physical device a score, and pick the device with the
        // lowest ("best") score.
        //
        // In this example, we simply select the best-scoring device to use in the application.
        // In a real-life setting, you may want to use the best-scoring device only as a
        // "default" or "recommended" device, and let the user choose the device themselves.
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("No suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            // A list of optional features and extensions that our program needs to work correctly.
            // Some parts of the Vulkan specs are optional and must be enabled manually at device
            // creation. In this example the only thing we are going to need is the `khr_swapchain`
            // extension that allows us to draw to a window.
            enabled_extensions: device_extensions,
            enabled_features: enabled_features,

            // The list of queues that we are going to use. Here we only use one queue, from the
            // previously chosen queue family.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let mut window_size = [0, 0];

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let composite_alpha = surface_capabilities.supported_composite_alpha.iter().next().unwrap();
        
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        window.set_cursor_visible(false);
        window_size = window.inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: window_size,
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha: composite_alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec3 position;
                layout(location = 1) in float intensity;
                layout(location = 2) in float index;
                layout(location = 0) out float f_intensity;
                layout(location = 1) out float f_index;

				void main() {
                    f_intensity = intensity;
                    f_index = index;
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

                layout(location = 0) in float f_intensity;
                layout(location = 1) in float f_index;
				layout(location = 0) out vec4 f_color;

				void main() {
                    float distance_to_center = pow(250/max(f_index, 1), 0.2);
                    float distance_to_center_white = pow(250/max(f_index, 1), 0.2);
                    float center = max(
                                       max(
                                            1 - abs(f_index - floor(f_index) - 0.545) * 2,
                                            0
                                       )
                                       * (1/distance_to_center*10+1) - (1/distance_to_center*10 - 1),
                                       0
                                    )
                                    * pow(f_intensity, 0.4)
                                    ;
					f_color = vec4(
                        min(pow(f_intensity, 0.75)/distance_to_center*2, 1),
                        min(f_intensity/distance_to_center*3 - 1, f_intensity/distance_to_center*2),
                        min(f_intensity*f_intensity/distance_to_center*3 - 2, 1.0),
                        min(f_intensity/distance_to_center, 1) * center * 0.95
                    );
				}
			"
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
            .blend_alpha_blending()
            .line_width(3f32)
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .line_strip()
            //.line_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut fft = vec![0f32; 512];
    let mut circle_increment = -20.70003f32;
    // -138.29922

    event_loop.run(move |event, _, control_flow| {
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        match event {
            Event::WindowEvent { event, .. } => match event {

                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                WindowEvent::Resized(_) => {
                    recreate_swapchain = true;
                }

                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => {
                        circle_increment += y as f32 / 10f32;
                        println!("({})", circle_increment);
                    }
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        circle_increment += p.y as f32 / 100f32;
                        println!("({})", circle_increment);
                    },
                    _ => ()
                },

                WindowEvent::KeyboardInput { input, .. } => match (input.state, input.virtual_keycode) {
                    (ElementState::Released, Some(VirtualKeyCode::F11)) => {
                        match window.fullscreen() {
                            Some(_) => window.set_fullscreen(None),
                            None => window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                        }
                    },
                    (ElementState::Released, Some(VirtualKeyCode::Escape)) => {
                        window.set_fullscreen(None)
                    }
                    _ => ()
                }

                _ => ()

            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                if recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {e:?}"),
                        };

                    swapchain = new_swapchain;
                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
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

                match receiver.recv_timeout(Duration::from_millis(20)) {
                    Ok(chunk) => {
                        fft = chunk;
                    }
                    Err(err) => (),
                }
                
                let aspect = window_size[0] as f32 / window_size[1] as f32;

                let proj_view = {
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect,
                        0.01,
                        1.0,
                    );
                    let view = Matrix4::look_at_rh(
                        Point3::new(0.0, 0.0, 30.0),
                        Point3::new(50.0, 0.0, 0.0),
                        Vector3::new(0.0, 0.0, 1.0),
                    );
                    proj * view
                };
            
                let vertex_buffer = {
                    vulkano::impl_vertex!(Vertex, position, intensity, index);
                    
                    CpuAccessibleBuffer::from_iter(
                        &memory_allocator,
                        BufferUsage {
                            vertex_buffer: true,
                            ..BufferUsage::empty()
                        },
                        false,
                        fft
                            .iter()
                            .enumerate()
                            //.rev()
                            .map(|(i, x)| gen_vertex_quasar(proj_view, circle_increment, fft.len(), i, *x) )
                            .collect::<Vec<Vertex>>()
                            .iter()
                            .cloned(),
                    )
                    .unwrap()
                };            

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_num as usize].clone(),
                            )
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
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
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_num)
                    )
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


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
    intensity: f32,
    index: f32
}


fn gen_vertex_flatvis(len: usize, i: usize, value: f32) -> [f32; 3] {
    [(i as f32).log(1.1f32) / (len as f32).log(1.1f32) * 2f32 - 1f32,
     - value/255f32 + 0.5f32,
     value/255f32]
}


fn gen_vertex_quasar(proj_view: Matrix4::<f32>, circle_increment: f32, len: usize, i: usize, value: f32) -> Vertex {
    let len = len as f32;
    let i = i as f32;
    let progress = i/len;

    let circle_progress = i * PI*2f32 * (0.5f32 - 1f32 / 12f32 + 0.01f32 + circle_increment/2000.0);
    let (x, y) = (circle_progress.sin(), circle_progress.cos());
    let asdf = [
        x / 10f32 + x * progress.powf(0.8f32) * (10f32) + 50.0,
        y / 10f32 + y * progress.powf(0.8f32) * (10f32),
        0.0, // (i/len).powf(1.1f32) * ((i % 2.0) as f32 * 2.0 - 1.0),
        1f32
    ];

    let mut result: [f32; 3] = (proj_view * Vector4::from(asdf)).truncate().into();
    // result[2] = 0f32; //if i < 12f32 { 0f32 } else { (value/60f32).min(1f32) };
    result[2] = if i < 12f32 { 0f32 } else { (value/60f32).min(1f32) };
    Vertex {
        position: result,
        intensity: if i < 12f32 { 0f32 } else { (value/50f32).powf(0.95f32).min(1f32) },
        index: i
    }
}


fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions();

    viewport.origin = [0.0, 0.0];
    viewport.dimensions = [dimensions.width() as f32, dimensions.height() as f32];
    viewport.depth_range = 0.0..1.0;

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

