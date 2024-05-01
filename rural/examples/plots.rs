use plotters::prelude::*;
use rand_core::SeedableRng;
use rand_distr::Distribution;
use rural::activation::{ktanh, pade, schraudolph, schraudolph_ng, spline, taylor};
use std::cmp::Ordering;
use std::error::Error;
use std::time::Instant;

type Editor<'s> = DrawingArea<SVGBackend<'s>, plotters::coord::Shift>;

// TODO(toms): normalize and improve colors

fn main() -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new("target/plots-approx.svg", (800, 800)).into_drawing_area();

    let (upper, lower) = root.split_vertically(400);
    {
        let (left, right) = upper.split_horizontally(400);
        plot_exp(left)?;
        plot_tanh(right)?;
    }
    {
        let (left, right) = lower.split_horizontally(400);
        mse(left)?;
        compute(right)?;
    }

    root.present()?;

    Ok(())
}

fn plot_exp(area: Editor) -> Result<(), Box<dyn Error>> {
    let x_range = -6f32..6.;
    let x_axis = x_range.clone().step(0.05);

    let mut cc = ChartBuilder::on(&area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("exp(x)", ("sans-serif", 20))
        .build_cartesian_2d(x_range, -1f32..10.)?;

    let mut plot_fn = |fxn: fn(f32) -> f32, label, color| -> Result<(), Box<dyn Error>> {
        cc.draw_series(LineSeries::new(
            x_axis.values().map(|x| (x, fxn(x))),
            &color,
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        Ok(())
    };

    plot_fn(libm::expf, "libm::expf", RED)?;
    plot_fn(schraudolph::expf, "schraudolph::expf", CYAN)?;
    plot_fn(schraudolph_ng::expf, "schraudolph_ng::expf", BLUE)?;

    cc.configure_mesh().disable_mesh().draw()?;
    cc.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;
    Ok(())
}

fn plot_tanh(area: Editor) -> Result<(), Box<dyn Error>> {
    let x_range = -6f32..6.;
    let x_axis = x_range.clone().step(0.05);

    let mut cc = ChartBuilder::on(&area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("tanh(x)", ("sans-serif", 20))
        .build_cartesian_2d(x_range, -1.1f32..1.1)?;

    let mut plot_fn = |fxn: fn(f32) -> f32, label, color| -> Result<(), Box<dyn Error>> {
        cc.draw_series(LineSeries::new(
            x_axis.values().map(|x| (x, fxn(x))),
            &color,
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        Ok(())
    };

    plot_fn(libm::tanhf, "libm::tanhf", RED)?;
    plot_fn(schraudolph::tanhf, "schraudolph::tanhf", CYAN)?;
    plot_fn(schraudolph_ng::tanhf, "schraudolph_ng::tanhf", BLUE)?;
    plot_fn(spline::tanhf3, "spline::tanhf", GREEN)?;
    plot_fn(ktanh::tanhf, "ktanh::tanhf", RED)?;
    plot_fn(taylor::tanhf, "taylor::tanhf", BLACK)?;
    plot_fn(pade::tanhf, "pade::tanhf", MAGENTA)?;

    cc.configure_mesh().disable_mesh().draw()?;
    cc.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;
    Ok(())
}

const X_MAX: f32 = 8.;

fn mse(area: Editor) -> Result<(), Box<dyn Error>> {
    let x_range = -X_MAX..X_MAX;
    let x_axis = x_range.clone().step(0.05);

    let y_max = 0.2f32;
    let mut cc = ChartBuilder::on(&area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("`tanh` accuracy", ("sans-serif", 20))
        .build_cartesian_2d(x_range, (0f32..y_max).log_scale())?;

    let mut plot_fn = |fxn: fn(f32) -> f32, label, color| -> Result<(), Box<dyn Error>> {
        let f = |x| (libm::tanhf(x) - fxn(x)).abs().max(f32::EPSILON); // max(f32::EPSILON) fixes 'explosions' in chart
        cc.draw_series(LineSeries::new(x_axis.values().map(|x| (x, f(x))), &color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        Ok(())
    };

    plot_fn(taylor::tanhf, "taylor", BLACK)?;
    plot_fn(schraudolph::tanhf, "schraudolph", CYAN)?;
    plot_fn(schraudolph_ng::tanhf, "schraudolph-ng", BLUE)?;
    plot_fn(spline::tanhf3, "spline", GREEN)?;
    plot_fn(ktanh::tanhf, "ktanh", RED)?;
    plot_fn(pade::tanhf, "pade", MAGENTA)?;

    cc.configure_mesh()
        .disable_mesh()
        // .x_desc("x")
        .y_desc("MSE")
        .draw()?;
    cc.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    Ok(())
}

fn compute(area: Editor) -> Result<(), Box<dyn Error>> {
    const NUM_STEPS: usize = 50_000_000;
    const N: f32 = NUM_STEPS as f32;

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
    let dist = rand_distr::Uniform::new(-X_MAX, X_MAX);

    let mut points = vec![];

    let control_duration = {
        let t0 = Instant::now();

        let mut mse = 0.;
        for _ in 0..NUM_STEPS {
            let x = dist.sample(&mut rng);
            mse += (libm::tanhf(x) - libm::tanhf(x)).powi(2) / N;
        }

        // println!("{}: mse={} duration={:?}ns", name, mse, duration);

        let name = "libm";

        let duration = Instant::now().duration_since(t0).as_nanos() / 2;

        let single_duration = duration as f32 / N;
        let speedup = 1.;

        println!("{}, {}, {:?}, {}", name, mse, single_duration, speedup);

        points.push((name, mse, speedup));

        duration
    };

    for (name, fxn) in [
        // ("libm", libm::tanhf as fn(f32) -> f32),
        ("taylor", taylor::tanhf as fn(f32) -> f32),
        ("schraudolph", schraudolph::tanhf as fn(f32) -> f32),
        ("schraudolph-ng", schraudolph_ng::tanhf),
        ("spline", spline::tanhf3),
        ("ktanh", ktanh::tanhf),
        ("pade", pade::tanhf),
    ] {
        let t0 = Instant::now();

        let mut mse = 0.;
        for _ in 0..NUM_STEPS {
            let x = dist.sample(&mut rng);
            mse += (libm::tanhf(x) - fxn(x)).powi(2) / N;
        }

        let duration = Instant::now().duration_since(t0).as_nanos() - control_duration;

        let speedup = control_duration as f32 / duration as f32;
        println!(
            "{}, {}, {:?}, {}",
            name,
            mse * 1e5,
            duration as f32 / N,
            speedup
        );

        points.push((name, mse, speedup));
    }

    let max_y = points
        .iter()
        .map(|(_, _, speedup)| *speedup)
        .max_by(|x, y| {
            if x > y {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        })
        .unwrap()
        .ceil();

    let mut cc = ChartBuilder::on(&area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("`tanh` performance", ("sans-serif", 20))
        .build_cartesian_2d((0f32..0.001).log_scale(), 1f32..max_y)?;

    points.sort_by(|(_, x, _), (_, y, _)| {
        if x > y {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    });

    // Draw dashed line for Pareto curve
    cc.draw_series(DashedLineSeries::new(
        points.iter().filter_map(|(name, mse, speedup)| {
            // Hard-coded filter for dominating points
            if ["libm", "pade", "schraudolph-ng", "schraudolph", "taylor"].contains(name) {
                Some((*mse, *speedup))
            } else {
                None
            }
        }),
        4,
        4,
        RGBColor(150, 150, 150).into(),
    ))?;

    cc.draw_series(PointSeries::of_element(
        points.into_iter(),
        8,
        ShapeStyle::from(&RGBColor(180, 180, 180)).filled(),
        &|(name, mse, speedup), size, style| {
            EmptyElement::at((mse, speedup))
                + Circle::new((0, 0), size, style)
                + Text::new(name, (-20, -17), ("sans-serif", 10))
        },
    ))?;

    cc.configure_mesh()
        .disable_mesh()
        .x_desc("MSE")
        .y_desc("speedup")
        .draw()?;

    Ok(())
}
