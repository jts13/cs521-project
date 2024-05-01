use rural::matrix::Matrix;

fn main() {
    let a = Matrix::new(&[[1., 2.], [4., 5.], [7., 8.]]);
    println!(" a[2, 1]: {:?}", a[(2, 1)]);

    let a = Matrix::new(&[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]);
    let b = Matrix::new(&[[6., 5., 4.], [9., 8., 7.], [3., 2., 1.]]);
    println!(" mm: {:?}", a.matmul(&b));
    println!("rmm: {:?}", a.rand_matmul(&b, 1.));

    let a = Matrix::new(&[[1., 2.], [4., 5.], [7., 8.]]);
    let b = Matrix::new(&[[6., 5., 4.], [3., 2., 1.]]);
    println!(" mm: {:?}", a.matmul(&b));
    println!("rmm: {:?}", a.rand_matmul(&b, 0.5));

    let a = Matrix::new(&[[1., 2.], [4., 5.], [7., 8.], [7., 8.]]);
    let b = Matrix::new(&[[6., 5., 4., 3.], [3., 2., 1., 0.]]);
    println!(" mm: {:?}", a.matmul(&b));
    println!("rmm: {:?}", a.rand_matmul(&b, 0.5));

    // let (m, n, k) = (50, 100, 80);
    let (m, n, k) = (5, 10, 8);
    let a = Matrix::random(m, n, 0);
    let b = Matrix::random(n, k, 0);
    println!(" mm: {:?}", a.matmul(&b));
    println!("rmm: {:?}", a.rand_matmul(&b, 0.5));
}
