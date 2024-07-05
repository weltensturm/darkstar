use core::panic;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use std::error;
use std::fs::read_to_string;
use std::io::Read;

use bytemuck::{Pod, Zeroable};
use vulkano::buffer::CpuBufferPool;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::memory::allocator::MemoryUsage;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::{Pipeline, PipelineBindPoint, StateMode};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, ColorSpace, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};

use cgmath::{Matrix4, Point3, Rad, Vector3, Vector4};
use std::convert::TryFrom;
use std::f32::consts::PI;
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
#[cfg(target_os = "windows")]
use winit::platform::windows::WindowBuilderExtWindows;
use winit::window::{Fullscreen, Window, WindowBuilder};

use crate::capture::CaptureCommand;

enum RenderMode {
    Quasar,
    Line,
    Circle,
}

impl RenderMode {
    fn next(&self) -> Self {
        use RenderMode::*;
        match *self {
            Quasar => Line,
            Line => Circle,
            Circle => Quasar,
        }
    }
}

struct VertexParams {
    mode: RenderMode,
    multiplier: f32,
    exponent: f32,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Globals {
    diminish_red: f32,
    diminish_green: f32,
    diminish_blue: f32,
}

pub fn window_loop(
    receiver: Receiver<Vec<f32>>,
    capture_command: SyncSender<CaptureCommand>,
) -> Result<(), Box<dyn error::Error>> {
    let surface_formats = vec![
        (Format::A2B10G10R10_UNORM_PACK32, ColorSpace::Hdr10St2084),
        (Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
    ];

    let hdr_colorspaces = [
        ColorSpace::Hdr10St2084
    ];

    let library = VulkanLibrary::new().expect("Vulkan");
    let required_extensions = vulkano::instance::InstanceExtensions {
        ext_swapchain_colorspace: true,
        ..vulkano_win::required_extensions(&library)
    };
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
    #[cfg(target_os = "windows")]
    {
        builder = builder.with_drag_and_drop(false); // otherwise conflicts with WASAPI
    }
    let event_loop = EventLoop::new();
    let surface = builder
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let enabled_features = Features {
        wide_lines: true,
        ..Features::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.graphics
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                        && p.supported_features().contains(&enabled_features)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
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
            enabled_extensions: device_extensions,
            enabled_features: enabled_features,
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

    let (mut swapchain, images, colorspace) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let composite_alpha = surface_capabilities
            .supported_composite_alpha
            .iter()
            .next()
            .unwrap();

        let (image_format, colorspace) = {
            for format in device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()
            {
                println!("{:?} {:?}", format.0, format.1);
            }

            let supported = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap();

            surface_formats
                .into_iter()
                .filter(|a| supported.contains(&a))
                .next()
                .expect("Surface not supported")
        };

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        window.set_cursor_visible(false);
        window_size = window.inner_size().into();

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format: Some(image_format),
                image_color_space: colorspace,
                image_extent: window_size,
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha: composite_alpha,
                present_mode: vulkano::swapchain::PresentMode::Fifo,
                ..Default::default()
            },
        )
        .expect("swapchain");

        (swapchain, images, colorspace)
    };

    let is_hdr = hdr_colorspaces.contains(&colorspace);

    let vs = {
        let file = read_to_string("shaders/16lines.vs").expect("16lines.vs");

        let mut spirv = glsl_to_spirv::compile(&file, glsl_to_spirv::ShaderType::Vertex)?;
        let mut spirv_bytes = vec![];
        spirv.read_to_end(&mut spirv_bytes)?;

        unsafe { ShaderModule::from_bytes(device.clone(), &spirv_bytes) }?
    };

    let fs = {
        let file = read_to_string("shaders/16lines.fs").expect("16lines.fs");

        let mut spirv = glsl_to_spirv::compile(&file, glsl_to_spirv::ShaderType::Fragment)?;
        let mut spirv_bytes = vec![];
        spirv.read_to_end(&mut spirv_bytes)?;

        unsafe { ShaderModule::from_bytes(device.clone(), &spirv_bytes) }?
    };

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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(
        device.clone(),
        &vs,
        &fs,
        &images,
        render_pass.clone(),
        &mut viewport,
    );
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut fft = vec![0f32; 512];
    let mut circle = VertexParams {
        mode: RenderMode::Quasar,
        // multiplier: 1.0023009,
        // multiplier: 1.0034024,
        multiplier: 0.98540187,
        exponent: 1f32,
    };
    // -138.29922

    let mut in_shift = false;
    let mut in_ctrl = false;

    let mut last_cursor_moved = SystemTime::now();

    let uniform_buffer = CpuBufferPool::<Globals>::new(
        memory_allocator.clone(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        MemoryUsage::Upload,
    );

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

                WindowEvent::CursorMoved { .. } => {
                    window.set_cursor_visible(true);
                    last_cursor_moved = SystemTime::now();
                }

                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => {
                        if in_ctrl {
                            circle.exponent += y as f32 / (if in_shift { 1000.0 } else { 100.0 });
                            println!("({})", circle.exponent);
                        } else {
                            circle.multiplier +=
                                y as f32 / (if in_shift { 40000.0 } else { 2000.0 });
                            println!("({})", circle.multiplier);
                        }
                    }
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        circle.exponent += p.y as f32 / 100f32;
                        println!("({})", circle.exponent);
                    }
                },

                WindowEvent::KeyboardInput { input, .. } => {
                    match (input.state, input.virtual_keycode) {
                        (ElementState::Released, Some(VirtualKeyCode::F11)) => {
                            match window.fullscreen() {
                                Some(_) => window.set_fullscreen(None),
                                None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
                            }
                        }
                        (ElementState::Released, Some(VirtualKeyCode::Escape)) => {
                            window.set_fullscreen(None)
                        }
                        (ElementState::Released, Some(VirtualKeyCode::T)) => {
                            circle.mode = circle.mode.next();
                        }
                        (ElementState::Released, Some(VirtualKeyCode::Space)) => {
                            capture_command
                                .send(CaptureCommand::SwitchCaptureType)
                                .unwrap();
                        }
                        (state, Some(VirtualKeyCode::LShift)) => {
                            in_shift = state == ElementState::Pressed;
                        }
                        (state, Some(VirtualKeyCode::LControl)) => {
                            in_ctrl = state == ElementState::Pressed;
                        }
                        _ => (),
                    }
                }

                _ => (),
            },
            Event::RedrawEventsCleared => {
                if last_cursor_moved != SystemTime::UNIX_EPOCH
                    && last_cursor_moved + Duration::from_secs(2) < SystemTime::now()
                {
                    last_cursor_moved = SystemTime::UNIX_EPOCH;
                    window.set_cursor_visible(false);
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                if recreate_swapchain {
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
                    (pipeline, framebuffers) = window_size_dependent_setup(
                        device.clone(),
                        &vs,
                        &fs,
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
                    let proj =
                        cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect, 0.01, 1.0);
                    let view = Matrix4::look_at_rh(
                        Point3::new(0.0, 0.0, 30.0),
                        Point3::new(50.0, 0.0, 0.0),
                        Vector3::new(0.0, 0.0, 1.0),
                    );
                    proj * view
                };

                let subbuffer = uniform_buffer
                    .from_data(Globals {
                        diminish_red: if is_hdr { 0.5 } else { 0.5 },
                        diminish_green: if is_hdr { 1.0/3.0 } else { 1.0/2.0 },
                        diminish_blue: if is_hdr { 1.0/8.0 } else { 1.0/3.0 },
                    })
                    .unwrap();

                let layout = pipeline.layout().set_layouts().get(0).expect("expected set layout 0");
                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, subbuffer)],
                )
                .unwrap();

                let vertex_buffer = {
                    vulkano::impl_vertex!(Vertex, position, intensity, index, mode);

                    CpuAccessibleBuffer::from_iter(
                        &memory_allocator,
                        BufferUsage {
                            vertex_buffer: true,
                            ..BufferUsage::empty()
                        },
                        false,
                        fft.iter()
                            .enumerate()
                            .map(|(i, x)| gen_vertex(proj_view, &circle, fft.len(), i, *x))
                            .collect::<Vec<Vertex>>(),
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
                    .set_line_width(dimensions.width as f32 / 1000.0)
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set,
                    )
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
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_num),
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
    index: f32,
    mode: i32,
}

fn gen_vertex_flatvis(len: usize, i: usize, value: f32) -> [f32; 3] {
    [
        (i as f32).log(1.1f32) / (len as f32).log(1.1f32) * 2f32 - 1f32,
        -value / 255f32 + 0.5f32,
        value / 255f32,
    ]
}

fn gen_vertex(
    proj_view: Matrix4<f32>,
    circle: &VertexParams,
    len: usize,
    i: usize,
    value: f32,
) -> Vertex {
    let len = len as f32;
    let i = i as f32;
    let progress = i / len;

    match circle.mode {
        RenderMode::Circle => {
            let value = value * (i / len).sqrt();
            // let circle_progress = i.powf(1.0/(circle.exponent/20.0)) * PI*2f32 * 85.0/2000.0 * circle.multiplier;
            let circle_progress =
                (progress.powf(circle.exponent) * PI * 2f32 * circle.multiplier) * len;
            // let (y, x) = (1f32/(1f32+i/len).log(2f32) + circle_increment/20.0, -value/50f32);
            let (x, y) = (circle_progress.sin(), circle_progress.cos());
            let flipflop = 1f32 - (i % 2f32) * 2f32;
            let expand = (value / 50.0).powf(2f32) * flipflop / (10.0 + progress.powf(0.8f32));
            let asdf = [
                x / 10f32 + x * expand + x * progress.powf(0.8f32) * (1.5f32) + 50.0,
                y / 10f32 + y * expand + y * progress.powf(0.8f32) * (1.5f32),
                0f32, // -value / 1000.0 * (1f32 - (i % 2f32)*2f32),
                1f32,
            ];

            let mut result: [f32; 3] = (proj_view * Vector4::from(asdf)).truncate().into();
            result[2] = 0f32; //if i < 12f32 { 0f32 } else { (value/60f32).min(1f32) };
                              // result[2] = value/60f32;
            Vertex {
                position: result,
                intensity: if i % 440f32 < 4f32 {
                    1f32
                } else {
                    (value / 2.0).powf(0.95f32).min(1f32)
                },
                index: i,
                mode: 0,
            }
        }
        RenderMode::Quasar => {
            let circle_progress = i
                * PI
                * 2f32
                * (0.5f32 - 1f32 / 12f32 + 0.01f32 + (-20.70003f32 * circle.multiplier) / 2000.0);
            let (x, y) = (circle_progress.sin(), circle_progress.cos());
            let asdf = [
                x / 10f32 + x * progress.powf(0.8f32) * (10f32) + 50.0,
                y / 10f32 + y * progress.powf(0.8f32) * (10f32),
                0.0, // (i/len).powf(1.1f32) * ((i % 2.0) as f32 * 2.0 - 1.0),
                1f32,
            ];

            let mut result: [f32; 3] = (proj_view * Vector4::from(asdf)).truncate().into();
            // result[2] = 0f32; //if i < 12f32 { 0f32 } else { (value/60f32).min(1f32) };
            result[2] = if i < 12f32 {
                0f32
            } else {
                (value / 60f32).min(1f32)
            };
            Vertex {
                position: result,
                intensity: if i < 12f32 {
                    0f32
                } else {
                    (value / 40f32).powf(0.95f32).min(1f32)
                },
                index: i,
                mode: 1,
            }
        }
        RenderMode::Line => {
            // let value = value*(i/len).sqrt();
            let (x, y) = (
                -value / 50.0,
                // progress.powf(circle.exponent)*circle.multiplier*2.0
                2595.0 * (i * 0.005).log10() / 4000.0,
            );

            let flipflop = 1f32; //1f32 - (i % 2f32)*2f32;

            let vec = [
                1.0 + x * (1.5f32) * flipflop + 50.0,
                1.0 - y * (1.5f32),
                0f32, // -value / 1000.0 * (1f32 - (i % 2f32)*2f32),
                1f32,
            ];
            let mut result: [f32; 3] = (proj_view * Vector4::from(vec)).truncate().into();
            result[2] = 0f32; //if i < 12f32 { 0f32 } else { (value/60f32).min(1f32) };
            Vertex {
                position: result,
                // intensity:
                //     if i % 440f32 < 4f32
                //         { 1f32 }
                //         else { (value/4.0).powf(0.95f32).min(1f32) },
                intensity: if i < 12f32 {
                    0f32
                } else {
                    (value / 40f32).powf(0.95f32).min(1f32)
                },
                index: i,
                mode: 0,
            }
        }
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
    let dimensions = images[0].dimensions();

    viewport.origin = [0.0, 0.0];
    viewport.dimensions = [dimensions.width() as f32, dimensions.height() as f32];
    viewport.depth_range = 0.0..1.0;

    let framebuffers = images
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
        .collect::<Vec<_>>();

    let pipeline = GraphicsPipeline::start()
        // .blend_alpha_blending()
        .rasterization_state(RasterizationState {
            line_width: StateMode::Dynamic,
            ..Default::default()
        })
        // .color_blend_state(ColorBlendState::new(1).blend_alpha())
        .color_blend_state(ColorBlendState::new(1).blend_additive())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
        // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineList))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    (pipeline, framebuffers)
}
