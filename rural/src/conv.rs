pub fn conv1d(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let output_size = input.len() - kernel.len() + 1;

    (0..output_size)
        .map(|i| crate::math::inner_product(&input[i..], kernel, 0.))
        .collect()
}

pub mod perforation {
    use rand_core::SeedableRng;
    use rand_distr::Distribution;

    fn find_index_of_nearest_neighbor<T: PartialEq>(
        arr: &[T],
        index: usize,
        value: T,
    ) -> Option<usize> {
        let mut left = index;
        let mut right = index;

        while left > 0 || right < arr.len() - 1 {
            if left > 0 {
                if arr[left - 1] == value {
                    return Some(left - 1);
                }
                left -= 1;
            }

            if right < arr.len() - 1 {
                if arr[right + 1] == value {
                    return Some(right + 1);
                }
                right += 1;
            }
        }

        None
    }

    fn pseudo_mask(length: usize) -> (Box<[bool]>, Box<[(usize, usize)]>) {
        let alpha = 1.5;
        let uniform_range = (0., 1.);

        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
        let uniform = rand_distr::Uniform::new(uniform_range.0, uniform_range.1);

        let mut sequence = vec![false; length];
        for a in (0..)
            .map(|i| {
                let i = i as f32;
                let u = uniform.sample(&mut rng);
                (alpha * (i + u)).ceil() as usize
            })
            .take_while(|&a| a < length)
        {
            sequence[a] = true;
        }

        let neighbors = (0..length)
            .filter(|&i| !sequence[i])
            .map(|i| {
                let neighbor = find_index_of_nearest_neighbor(&sequence, i, true).unwrap();
                (i, neighbor)
            })
            .collect();

        (sequence.into_boxed_slice(), neighbors)
    }

    pub fn conv1d(input: &[f32], kernel: &[f32]) -> Vec<f32> {
        let output_size = input.len() - kernel.len() + 1;

        let (mask, neighbors) = pseudo_mask(output_size);

        debug_assert_eq!(mask.len(), output_size);

        let mut output: Vec<_> = (0..output_size)
            .map(|i| {
                if mask[i] {
                    crate::math::inner_product(&input[i..], kernel, 0.)
                } else {
                    0. // placeholder - fill in with `neighbors`
                }
            })
            .collect();

        for &(i, j) in neighbors.iter() {
            output[i] = output[j];
        }

        output
    }

    #[test]
    fn baseline() {
        let input: Box<[_]> = (0..20).map(|i| i as f32 + 1.).collect();
        let kernel: Box<[_]> = (0..3).map(|_| 1.).collect();

        let result = super::conv1d(&input, &kernel);
        println!("conv1d: {result:?}");
        assert_eq!(
            result.as_slice(),
            &[
                6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 51., 54.,
                57.
            ]
        );

        // let mask_size = input.len() - kernel.len() + 1;
        // let mask = pseudo_mask(mask_size);
        // println!("mask[{mask_size}]: {mask:?}",);

        let result = conv1d(&input, &kernel);
        println!("perforation::conv1d: {result:?}");
        assert_eq!(
            result.as_slice(),
            &[
                9., 9., 9., 15., 18., 21., 21., 27., 30., 30., 39., 39., 42., 45., 48., 48., 54.,
                57.
            ]
        );
    }
}
