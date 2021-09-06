use std::collections::VecDeque;
use std::error;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread;
use std::time::{Duration, Instant};
use wasapi::*;
use rustfft::{FftPlanner, num_complex::Complex};
use crossterm::{QueueableCommand, cursor, terminal::size};
use std::io::{Write, stdout};
mod window;


#[macro_use]
extern crate log;
use simplelog::*;


const UPPER_BOUND: usize = 4096*16;


fn main() -> Res<()> {
    let _ = SimpleLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            .set_time_format_str("%H:%M:%S%.3f")
            .build(),
    );

    initialize_mta()?;

    let (tx_capt, rx_capt): (
        SyncSender<Vec<u8>>,
        Receiver<Vec<u8>>,
    ) = sync_channel(1);

    let (gfx_sender, gfx_receiver) = sync_channel(1);

    let chunksize = 4096*2;

    let _ = thread::Builder::new()
        .name("Capture".to_string())
        .spawn(move || {
            let result = capture_loop(tx_capt, chunksize);
            if let Err(err) = result {
                error!("Capture failed with error {}", err);
            }
        });

    let _ = thread::Builder::new()
        .name("FFT".to_string())
        .spawn(move || {
            fft_loop(rx_capt, gfx_sender);
        });

    window::window_loop(gfx_receiver);

    Ok(())
    
}


fn print_bars(buffer: Vec<f32>){
    let mut stdout = stdout();
    stdout.queue(cursor::Hide).map_err(|err| error!("{}", err)).ok();
    stdout.queue(cursor::MoveTo(0, 0)).map_err(|err| error!("{}", err)).ok();
    if let Ok((w, h)) = size() {
        let bass_cutoff = 12;
        let log_base = (UPPER_BOUND as f32/4.0f32).powf(1.0f32 / (w as f32 + bass_cutoff as f32));
        for y in 0..h-1 {
            let mut line = vec![' '; w as usize];
            for x in 0..w {
                if log_base.powi(x as i32 + bass_cutoff) as usize == log_base.powi(x as i32+bass_cutoff+1) as usize {
                    continue;
                }
                let max =
                    buffer[log_base.powi(x as i32 + bass_cutoff) as usize .. log_base.powi(x as i32+bass_cutoff+1) as usize]
                    .iter()
                    .fold(0f32, |a, &b| a.max(b.abs()));
                line[x as usize] =
                    if max > (h as f32 - 1f32 - y as f32)*8f32 { '|' } else { ' ' };
            }
            stdout.write(line.iter().collect::<String>().as_bytes()).map_err(|err| error!("{}", err)).ok();
        }
    }
    stdout.flush().map_err(|err| error!("{}", err)).ok();
}


fn fft_loop(raw: Receiver<Vec<u8>>, transformed: SyncSender<Vec<f32>>){
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(UPPER_BOUND);

    let mut buffer = vec![Complex{ re: 0f32, im: 0f32 }; UPPER_BOUND];

    let mut frame_start = Instant::now();

    loop {
        match raw.recv() {
            Ok(chunk) => {
                let floats: Vec::<f32>;
                unsafe {
                    let (_, floats_tmp, _) = chunk.align_to::<f32>();
                    floats = floats_tmp.to_vec();
                }
                buffer.splice(0..floats.len(), floats.clone().into_iter().map(|x| Complex::new(x, 0.0)).collect::<Vec<Complex<f32>>>());
                buffer.splice(floats.len()..buffer.len(), (floats.len()..buffer.len()).into_iter().map(|_| Complex::new(0.0, 0.0)).collect::<Vec<Complex<f32>>>());
            }
            Err(err) => error!("Some error {}", err),
        }
        
        if frame_start + Duration::from_millis(6) < Instant::now() {
            frame_start = Instant::now();
            fft.process(&mut buffer);
            //print_bars(buffer.iter().map(|x| x.to_polar().0).collect::<Vec<f32>>());
            let mut half = buffer.iter().map(|x| x.to_polar().0).collect::<Vec<f32>>();
            half = half[0..half.len()/2].to_vec();
            transformed.try_send(half).ok();
            print!(" {:?} ", Instant::now() - frame_start);
        }
    }
}


type Res<T> = Result<T, Box<dyn error::Error>>;


fn capture_loop(tx_capt: SyncSender<Vec<u8>>, chunksize: usize) -> Res<()> {
    let device = get_default_device(&Direction::Render)?;
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
        if h_event.wait_for_event(1000000).is_err() {
            error!("error, stopping capture");
            audio_client.stop_stream()?;
            break;
        }
    }
    Ok(())
}
