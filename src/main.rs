
use std::error;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread;
use std::time::{Duration, Instant};
use rustfft::{FftPlanner, num_complex::Complex};
use crossterm::{QueueableCommand, cursor, terminal::size};
use std::io::{Write, stdout};
mod window;


mod capture;
use capture::capture_loop;

use crate::capture::CaptureCommand;


// use windows::core::Error as WindowsError;
// use windows::Win32::System::Power::{
//     SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED, EXECUTION_STATE,
// };


#[macro_use]
extern crate log;
use simplelog::*;


const FFT_INPUT_SIZE: usize = 2usize.pow(10);
const FFT_RESULT_SIZE: usize = 2usize.pow(16);


type Res<T> = Result<T, Box<dyn error::Error>>;


fn main() -> Res<()> {
    let _ = SimpleLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            .set_time_format_str("%H:%M:%S%.3f")
            .build(),
    );

    let (to_fft, from_audio) = sync_channel::<(usize, u16, Vec<f32>)>(1);

    let (to_capture_command, capture_commands) = sync_channel::<CaptureCommand>(1);

    let (to_gpu, from_fft) = sync_channel(1);

    let _ = thread::Builder::new()
        .name("Capture".to_string())
        .spawn(move || {
            let result = capture_loop(to_fft, capture_commands);
            if let Err(err) = result {
                error!("Capture failed with error {}", err);
            }
        });

    let _ = thread::Builder::new()
        .name("FFT".to_string())
        .spawn(move || {
            fft_loop(from_audio, to_gpu).unwrap();
        });

    window::window_loop(from_fft, to_capture_command).unwrap();

    Ok(())
    
}


fn _print_bars(buffer: Vec<f32>){
    let mut stdout = stdout();
    stdout.queue(cursor::Hide).map_err(|err| error!("{}", err)).ok();
    stdout.queue(cursor::MoveTo(0, 0)).map_err(|err| error!("{}", err)).ok();
    if let Ok((w, h)) = size() {
        let bass_cutoff = 12;
        let log_base = (FFT_RESULT_SIZE as f32/4.0f32).powf(1.0f32 / (w as f32 + bass_cutoff as f32));
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


fn fft_loop(from_audio: Receiver<(usize, u16, Vec<f32>)>, to_gpu: SyncSender<Vec<f32>>) -> Res<()> {

    type Number = f64;
    let mut planner = FftPlanner::<Number>::new();
    let fft = planner.plan_fft_forward(FFT_RESULT_SIZE);

    let mut frame: Vec::<Number> = vec![];
    let mut buffer = vec![Complex{ re: 0 as Number, im: 0 as Number }; FFT_RESULT_SIZE];
    let mut scratch = vec![Complex{ re: 0 as Number, im: 0 as Number }; FFT_RESULT_SIZE];

    let mut frame_start = Instant::now();

    loop {
        match from_audio.recv() {
            Ok((sample_rate, channels, chunk)) => {

                let floats: Vec::<Number> =
                    frame
                        .into_iter()
                        .chain(chunk.into_iter().map(|x| x as Number))
                        .rev()
                        .take(sample_rate / 30)
                        .collect::<Vec<Number>>()
                        .into_iter()
                        .rev()
                        .collect();

                frame = floats;

                buffer.splice(
                    0 .. frame.len(),
                    frame
                        .iter()
                        .map(|x| Complex::new(*x*0.75, 0.0))
                        .collect::<Vec<Complex<Number>>>()
                );
                for v in &mut buffer[frame.len()..] {
                    *v = Complex::new(0.0, 0.0);
                }
            }
            Err(err) => {
                eprintln!("Some error {}", err);
                break Err(Box::new(err))
            }
        }
        
        if frame_start + Duration::from_millis(3) < Instant::now() {
            frame_start = Instant::now();
            fft.process_with_scratch(&mut buffer, &mut scratch);
            // _print_bars(buffer.iter().map(|x| x.to_polar().0).collect::<Vec<f32>>());
            let mut half = buffer.iter().map(|x| x.to_polar().0 as f32).collect::<Vec<f32>>();
            half = half[0..half.len()/2].to_vec();
            to_gpu.try_send(half).ok();
            // print!("FFT {:?}\n", Instant::now() - frame_start);
        }
    }
}

