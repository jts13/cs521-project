use rural::matrix::Matrix;

#[derive(Debug)]
struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::random(input_size, output_size, 0);
        let biases = Matrix::random(1, output_size, 0);
        DenseLayer { weights, biases }
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        input.matmul(&self.weights).add(&self.biases)
    }
}

fn main() {
    let input_size = 2;
    let output_size = 3;

    let layer = DenseLayer::new(input_size, output_size);

    let input = Matrix::random(1, input_size, 0);
    assert_eq!(input.shape(), (1, input_size));

    let output = layer.forward(&input);
    assert_eq!(output.shape(), (1, output_size));

    println!("layer: {:?}", layer);
    println!("input: {:?}", input);
    println!("output: {:?}", output);
}

// <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[1.3372822, 2.3078232, 1.172242 ]], dtype=float32)>
//
// layer: DenseLayer { weights: Matrix { data: [0.7385979, 0.1139881, 0.611827, 0.2235182, 0.6560209, 0.16295183], shape: (2, 3) }, biases: Matrix { data: [0.15164787, 0.8817933, 0.23451138], shape: (1, 3) } }
// input: Matrix { data: [1.0, 2.0], shape: (1, 2) }
// output: Matrix { data: [1.3372822, 2.3078232, 1.172242], shape: (1, 3) }
