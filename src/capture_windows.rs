
use std::error;
use std::collections::VecDeque;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel, TryRecvError};
use std::thread::sleep;
use core::time::Duration;

// use wasapi::*;

use cpal::{
    StreamError,
    SampleRate,
    traits::{HostTrait, DeviceTrait, StreamTrait}
};

// use windows::core::Error as WindowsError;
// use windows::Win32::System::Power::{
//     SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED, EXECUTION_STATE,
// };

type Res<T> = Result<T, Box<dyn error::Error>>;


// pub fn capture_loop_old(tx_capt: SyncSender<(usize, Vec<f32>)>, chunksize: usize) -> Res<()> {

//     initialize_mta()?;

//     unsafe {
//         let flags = ES_CONTINUOUS | ES_DISPLAY_REQUIRED;
//         if SetThreadExecutionState(flags) == EXECUTION_STATE(0) {
//             return Err(Box::new(WindowsError::from_win32()));
//         }
//     }

//     loop {
//         let device = get_default_device(&Direction::Render)?;
//         println!("Using audio device: {}", device.get_friendlyname().unwrap());
//         let mut audio_client = device.get_iaudioclient()?;

//         let channels = 1;

//         let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 44100, channels);

//         let blockalign = desired_format.get_blockalign();

//         let (_def_time, min_time) = audio_client.get_periods()?;

//         audio_client.initialize_client(
//             &desired_format,
//             min_time as i64,
//             &Direction::Capture,
//             &ShareMode::Shared,
//             true,
//         )?;

//         let h_event = audio_client.set_get_eventhandle()?;

//         let buffer_frame_count = audio_client.get_bufferframecount()?;

//         let render_client = audio_client.get_audiocaptureclient()?;
//         let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(
//             channels * blockalign as usize * buffer_frame_count as usize,
//         );
//         audio_client.start_stream()?;
//         loop {
//             if let Err(msg) = h_event.wait_for_event(1000) {
//                 eprintln!("{}", msg);
//                 audio_client.stop_stream()?;
//                 break;
//             }
//             while sample_queue.len() > chunksize {
//                 let mut chunk = vec![0u8; chunksize];
//                 for element in chunk.iter_mut() {
//                     *element = sample_queue.pop_front().unwrap();
//                 }
//                 let floats = unsafe {
//                     let (_, floats_tmp, _) = chunk.align_to::<f32>();
//                     floats_tmp.to_vec()
//                 };
//                 tx_capt.send((desired_format.get_samplespersec() as usize, floats))?;
//             }
//             render_client.read_from_device_to_deque(blockalign as usize, &mut sample_queue)?;
//         }
//     }
//     Ok(())
// }


enum Op {
    OutputChanged,
    Error(Res<()>)
}


pub enum CaptureCommand {
    SwitchCaptureType,

}

enum CaptureType {
    Input,
    Output,
}

impl CaptureType {
    fn next(&self) -> Self {
        match self {
            Self::Input => Self::Output,
            Self::Output => Self::Input,
        }
    }
}


pub fn capture_loop(to_fft: SyncSender<(usize, Vec<f32>)>, commands: Receiver::<CaptureCommand>) -> Res<()> {
    let host = cpal::default_host();

    let mut capture_type = CaptureType::Output;
    
    loop {

        let (device, config) = match capture_type {
            CaptureType::Input => {

                println!("{}", host.input_devices().unwrap().map(|a| a.name().unwrap() ).collect::<Vec<String>>().join("\n"));

                let device = host.default_input_device().unwrap();
        
                let mut supported_configs = device.supported_input_configs().unwrap();
                
                let config = supported_configs.next()
                    .expect("no supported config")
                    .with_max_sample_rate()
                    .config();
                
                (device, config)
    
            }
            CaptureType::Output => {

                println!("{}", host.output_devices().unwrap().map(|a| a.name().unwrap() ).collect::<Vec<String>>().join("\n"));

                let device = host.default_output_device().unwrap();
        
                let mut supported_configs = device.supported_output_configs().unwrap();
                
                let config = supported_configs.next()
                    .expect("no supported config")
                    .with_max_sample_rate()
                    .config();
        
                (device, config)
            }
        };

        let target_sample_rate = 30000;
        let mut sample_reduction: usize = 1;

        while config.sample_rate.0 as usize / sample_reduction > target_sample_rate {
            sample_reduction += 1;
        }

        let fake_sample_rate = config.sample_rate.0 as usize / sample_reduction;
        let sample_skip = config.channels as usize * sample_reduction;

        println!(
            "{} Audio Device: {} {} {} -> {} {:?}",
            match capture_type { CaptureType::Input => "Input", _ => "Output" },
            device.name().unwrap(),
            config.channels,
            config.sample_rate.0,
            fake_sample_rate,
            config.buffer_size
        );

        let to_fft = to_fft.clone();

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                to_fft.send((
                    fake_sample_rate,
                    data
                        .iter()
                        .step_by(sample_skip)
                        .copied()
                        .collect::<Vec<f32>>()
                )).unwrap();
            },
            move |err| {
                eprintln!("{}", err);
            },
            None
        ).unwrap();

        stream.play().unwrap();

        let result = loop {
            let default_device = match
                    match capture_type {
                        CaptureType::Input => host.default_input_device(),
                        CaptureType::Output => host.default_output_device()
                    }
                {
                Some(a) => a,
                None => { break Op::Error(Err("No default device".into())) }
            };

            if default_device.name().unwrap() != device.name().unwrap() {
                break Op::OutputChanged;
            } else {
                match commands.try_recv() {
                    Ok(CaptureCommand::SwitchCaptureType) => {
                        capture_type = capture_type.next();
                        break Op::OutputChanged;
                    },
                    Err(TryRecvError::Empty) => {}
                    Err(e) => break Op::Error(Err(e.into()))
                }
                sleep(Duration::from_millis(12));
                continue;
            }
        };

        match result {
            Op::OutputChanged => continue,
            Op::Error(Err(e)) => return Err(e),
            _ => panic!("what")
        }

    }

}
