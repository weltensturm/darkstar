
use std::error;
use std::collections::VecDeque;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel, TryRecvError};
use std::thread::sleep;
use core::time::Duration;


use cpal::{
    StreamError,
    SampleRate,
    traits::{HostTrait, DeviceTrait, StreamTrait}
};


type Res<T> = Result<T, Box<dyn error::Error>>;


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


pub fn capture_loop(to_fft: SyncSender<(usize, u16, Vec<f32>)>, commands: Receiver::<CaptureCommand>) -> Res<()> {
    let host = cpal::default_host();

    let mut capture_type = CaptureType::Output;
    
    loop {

        let (device, config) = match capture_type {
            CaptureType::Input => {

                let device = host.default_input_device().unwrap();
        
                let mut supported_configs = device.supported_input_configs().unwrap();
                
                let config = supported_configs.next()
                    .expect("no supported config")
                    .with_max_sample_rate()
                    .config();
                
                (device, config)
    
            }
            CaptureType::Output => {

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

        let sample_reduction = {
            let mut sample_reduction: usize = 1;
            while config.sample_rate.0 as usize / sample_reduction > target_sample_rate {
                sample_reduction += 1;
            }
            sample_reduction
        };

        let fake_sample_rate = config.sample_rate.0 as usize / sample_reduction;
        let sample_skip = config.channels as usize * sample_reduction;
        let channels = config.channels;

        println!(
            "Audio Device: {} {} {} -> {} {:?}",
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
                    channels,
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
