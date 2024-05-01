use plotters::prelude::*;
use rural::matrix::Matrix;
use std::error::Error;
use std::time::{Duration, Instant};

fn mean_squared_error(actual: &Matrix, predicted: &Matrix) -> f32 {
    assert_eq!(actual.shape(), predicted.shape());

    let mut mse = 0.;

    let (m, n) = actual.shape();

    for i in 0..m {
        for j in 0..n {
            let error = actual[(i, j)] - predicted[(i, j)];
            mse += error * error;
        }
    }

    let num_elements = m * n;
    mse / (num_elements as f32)
}

fn timed<R>(f: impl Fn() -> R) -> (R, Duration) {
    const I: u64 = 16;

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
    let root = SVGBackend::new("target/plots-matmul.svg", (800, 800)).into_drawing_area();

    let mut cc = ChartBuilder::on(&root)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("rand-matmul", ("sans-serif", 20))
        .build_cartesian_2d(0f32..1., 0f32..10.)?;

    for (size, color) in [
        (32, RGBColor(0, 0, 0)),
        (64, RGBColor(200, 0, 0)),
        (128, RGBColor(0, 200, 0)),
        (256, RGBColor(0, 0, 200)),
        (512, RGBColor(200, 100, 0)),
        (1024, RGBColor(0, 100, 200)),
    ] {
        let input_size = size;
        let output_size = size;

        let a = Matrix::random(input_size, output_size, 0);
        let b = Matrix::random(output_size, input_size, 0);

        let (expected, dt) = timed(|| a.matmul(&b));
        println!("M: size={size} dt={dt:?}");

        let mut measurements = vec![];

        let n = 100;
        for factor in [10, 25, 50, 75, 90, 95, 99, n] {
            let factor = factor as f32 / n as f32;

            let (actual, rdt) = timed(|| a.rand_matmul(&b, factor));

            let mse = mean_squared_error(&expected, &actual);

            println!("R: size={size} factor={factor:.2} mse={mse:.6} rdt={rdt:?}");

            let speedup = dt.as_secs_f32() / rdt.as_secs_f32();

            measurements.push((factor, mse, speedup));
        }

        {
            let (label, color, points) = (size.to_string(), color, &measurements);

            cc.draw_series(LineSeries::new(
                points
                    .iter()
                    .cloned()
                    .map(|(factor, _, speedup)| (factor, speedup)),
                &color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

            cc.draw_series(PointSeries::of_element(
                points.iter().cloned(),
                8,
                ShapeStyle::from(&color).filled(),
                &|(factor, mse, speedup), size, style| {
                    EmptyElement::at((factor, speedup))
                        + Circle::new((0, 0), size as f32 * mse.log10(), style)
                    // + Text::new(format!("{mse:.3}"), (0, 10), ("sans-serif", 10))
                },
            ))?;
        }
    }

    // Draw dashed line for 'baseline'
    cc.draw_series(DashedLineSeries::new(
        [(0., 1.), (1., 1.)],
        5,
        5,
        BLUE.into(),
    ))?;

    cc.configure_mesh()
        .disable_mesh()
        .x_desc("factor")
        .y_desc("speedup")
        .draw()?;
    cc.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;

    Ok(())
}
