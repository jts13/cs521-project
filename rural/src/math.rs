pub fn inner_product(x: &[f32], y: &[f32], init: f32) -> f32 {
    x.iter().zip(y).fold(init, |acc, (x, y)| acc + x * y)
}
