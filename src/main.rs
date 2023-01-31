
use std::error;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread;
use std::time::{Duration, Instant};
use rustfft::{FftPlanner, num_complex::Complex};
use crossterm::{QueueableCommand, cursor, terminal::size};
use std::io::{Write, stdout};
mod window;


#[cfg(target_os = "windows")]
mod capture_windows;
#[cfg(target_os = "windows")]
use capture_windows::capture_loop;

#[cfg(target_os = "linux")]
mod capture_linux;
#[cfg(target_os = "linux")]
use capture_linux::capture_loop;


#[macro_use]
extern crate log;
use simplelog::*;


const FFT_INPUT_SIZE: usize = 2usize.pow(11);
const FFT_RESULT_SIZE: usize = 2usize.pow(16);


type Res<T> = Result<T, Box<dyn error::Error>>;


fn main() -> Res<()> {
    let _ = SimpleLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            .set_time_format_str("%H:%M:%S%.3f")
            .build(),
    );

    let (tx_capt, rx_capt): (
        SyncSender<Vec<u8>>,
        Receiver<Vec<u8>>,
    ) = sync_channel(1);

    let (gfx_sender, gfx_receiver) = sync_channel(1);

    let chunksize = 1024;

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


fn fft_loop(raw: Receiver<Vec<u8>>, transformed: SyncSender<Vec<f32>>){
    loop {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_RESULT_SIZE);

        let mut frame: Vec::<f32> = vec![];
        let mut buffer = vec![Complex{ re: 0f32, im: 0f32 }; FFT_RESULT_SIZE];

        let mut frame_start = Instant::now();

        loop {
            match raw.recv() {
                Ok(chunk) => {
                    let floats = unsafe {
                        let (_, floats_tmp, _) = chunk.align_to::<f32>();
                        floats_tmp.to_vec()
                    };

                    let floats: Vec::<f32> =
                        frame
                            .into_iter()
                            .chain(floats.into_iter())
                            .rev()
                            .take(FFT_INPUT_SIZE)
                            .collect::<Vec<f32>>()
                            .into_iter()
                            .rev()
                            .collect();

                    frame = floats;

                    buffer.splice(
                        0 .. frame.len(),
                        frame
                            .clone()
                            .into_iter()
                            .rev()
                            .map(|x| Complex::new(x, 0.0))
                            .collect::<Vec<Complex<f32>>>()
                    );
                    buffer.splice(
                        frame.len()..buffer.len(),
                        (frame.len()..buffer.len())
                            .into_iter()
                            .map(|_| Complex::new(0.0, 0.0))
                            .collect::<Vec<Complex<f32>>>()
                    );
                }
                Err(err) => {
                    eprintln!("Some error {}", err);
                    break
                }
            }
            
            if frame_start + Duration::from_millis(3) < Instant::now() {
                frame_start = Instant::now();
                fft.process(&mut buffer);
                // _print_bars(buffer.iter().map(|x| x.to_polar().0).collect::<Vec<f32>>());
                let mut half = buffer.iter().map(|x| x.to_polar().0).collect::<Vec<f32>>();
                half = half[0..half.len()/2].to_vec();
                transformed.try_send(half).ok();
                print!("FFT {:?}\n", Instant::now() - frame_start);
            }
        }
    }
}

