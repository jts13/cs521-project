use plotters::prelude::*;
use rand_core::SeedableRng as _;
use rand_distr::Distribution as _;
use rural::conv::{conv1d, perforation};
use std::error::Error;
use std::time::{Duration, Instant};

fn mean_squared_error(actual: &[f32], predicted: &[f32]) -> f32 {
    assert_eq!(actual.len(), predicted.len());

    let mut mse = 0.;

    for i in 0..actual.len() {
        let error = actual[i] - predicted[i];
        mse += error * error;
    }

    mse / (actual.len() as f32)
}

fn timed<R>(f: impl Fn() -> R) -> (R, Duration) {
    const I: u64 = 10 * 1024;

    let mut i = 0;
    let mut dt = 0;

    loop {
        let t0 = Instant::now();
        let ret = f();

        dt += Instant::now().duration_since(t0).as_nanos();

        if i >= I {
            return (ret, Duration::from_nanos((dt / I as u128) as u64));
        }

        i += 1;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new("target/plots-conv.svg", (800, 800)).into_drawing_area();

    const MAX_X: f32 = 1400.;

    let mut cc = ChartBuilder::on(&root)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("Perforated `conv1d` (⍺ = 1.5)", ("sans-serif", 20))
        .build_cartesian_2d(0f32..MAX_X, 0f32..2.)?;

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
    let dist = rand_distr::StandardNormal;

    for (input_size, color) in [
        (32, RGBColor(100, 0, 0)),
        (64, RGBColor(200, 0, 0)),
        (128, RGBColor(0, 200, 0)),
        (256, RGBColor(0, 0, 200)),
        (512, RGBColor(200, 100, 0)),
        (768, RGBColor(0, 100, 200)),
        (1024, RGBColor(0, 120, 220)),
        (1280, RGBColor(0, 140, 240)),
        // (4096, RGBColor(0, 100, 200)),
    ]
    .into_iter()
    .rev()
    {
        let mut measurements = vec![];

        // TODO(toms): use random values for input
        let input: Box<[_]> = (0..input_size).map(|_| dist.sample(&mut rng)).collect();

        // for n in [4, 8, 16, 32, 64].iter().rev() {
        for n in (2..)
            .map(|n| 2usize.pow(n))
            .take_while(|n| (n * 2) <= input_size)
        {
            let kernel_size = input_size / n;

            // TODO(toms): use random values for kernel
            let kernel: Box<[_]> = (0..kernel_size).map(|_| dist.sample(&mut rng)).collect();

            let (expected, dt) = timed(|| conv1d(&input, &kernel));
            println!("M: size={input_size} dt={dt:?}");

            {
                let (actual, rdt) = timed(|| perforation::conv1d(&input, &kernel));

                let mse = mean_squared_error(&expected, &actual);

                let speedup = dt.as_secs_f32() / rdt.as_secs_f32();

                println!("R: input_size={input_size} kernel_size={kernel_size} mse={mse:.6} dt={dt:?} rdt={rdt:?} speedup={speedup:?}");

                measurements.push((input_size as f32, kernel_size as f32, mse, speedup));
            }
        }

        {
            let (label, color, points) = (input_size.to_string(), color, &measurements);

            cc.draw_series(LineSeries::new(
                points
                    .iter()
                    .cloned()
                    .map(|(input_size, _, _, speedup)| (input_size, speedup)),
                &color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

            // TODO(toms): change size of point based on MSE

            cc.draw_series(PointSeries::of_element(
                points.iter().cloned(),
                8,
                ShapeStyle::from(&color).filled(),
                &|(input_size, kernel_size, mse, speedup), size, style| {
                    EmptyElement::at((input_size, speedup))
                        + Circle::new((0, 0), size as f32 * mse.log10(), style)
                        + Text::new(format!("k={kernel_size}"), (10, 0), ("sans-serif", 10))
                },
            ))?;
        }
    }

    // Draw dashed line for 'baseline'
    cc.draw_series(DashedLineSeries::new(
        [(0., 1.), (MAX_X, 1.)],
        5,
        5,
        BLUE.into(),
    ))?;

    cc.configure_mesh()
        .disable_mesh()
        .x_desc("input_size")
        .y_desc("speedup")
        .draw()?;
    cc.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;

    Ok(())
}
