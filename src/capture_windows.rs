
use std::error;
use std::collections::VecDeque;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};

use wasapi::*;

use windows::core::Error as WindowsError;
use windows::Win32::System::Power::{
    SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED, EXECUTION_STATE,
};

type Res<T> = Result<T, Box<dyn error::Error>>;


pub fn capture_loop(tx_capt: SyncSender<Vec<u8>>, chunksize: usize) -> Res<()> {

    initialize_mta()?;

    unsafe {
        let flags = ES_CONTINUOUS | ES_DISPLAY_REQUIRED;
        if SetThreadExecutionState(flags) == EXECUTION_STATE(0) {
            // eprintln!();
            return Err(Box::new(WindowsError::from_win32()));
        }
    }

    loop {
        let device = get_default_device(&Direction::Render)?;
        println!("Using audio device: {}", device.get_friendlyname().unwrap());
        let mut audio_client = device.get_iaudioclient()?;

        let channels = 1;

        let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 44100, channels);

        let blockalign = desired_format.get_blockalign();

        let (_def_time, min_time) = audio_client.get_periods()?;

        audio_client.initialize_client(
            &desired_format,
            min_time as i64,
            &Direction::Capture,
            &ShareMode::Shared,
            true,
        )?;

        let h_event = audio_client.set_get_eventhandle()?;

        let buffer_frame_count = audio_client.get_bufferframecount()?;

        let render_client = audio_client.get_audiocaptureclient()?;
        let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(
            channels * blockalign as usize * buffer_frame_count as usize,
        );
        audio_client.start_stream()?;
        loop {
            while sample_queue.len() > chunksize {
                let mut chunk = vec![0u8; chunksize];
                for element in chunk.iter_mut() {
                    *element = sample_queue.pop_front().unwrap();
                }
                tx_capt.send(chunk)?;
            }
            render_client.read_from_device_to_deque(blockalign as usize, &mut sample_queue)?;
            if h_event.wait_for_event(1000).is_err() {
                eprintln!("error, stopping capture");
                audio_client.stop_stream()?;
                break;
            }
        }
    }
    Ok(())
}
