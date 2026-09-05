use num_bigint::BigUint;
use num_traits::{Zero, One};
use crate::ecc::Point;
use crate::jacobian::JacobianPoint;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuScalar {
    pub limbs: [u64; 4],
}

pub struct PollardKangaroo {
    pub start: BigUint,
    pub end: BigUint,
    pub target: Point,
}

impl PollardKangaroo {
    pub fn new(start: BigUint, end: BigUint, target: Point) -> Self {
        Self { start, end, target }
    }

    pub fn solve(&self, p: &BigUint, n: &BigUint) -> Option<BigUint> {
        // Fast CPU execution path until native CUDA kernels are bound directly
        let range = &self.end - &self.start;
        let max_steps = range.sqrt();

        let mut curr_point = JacobianPoint::from_affine(&self.target);
        let mut curr_scalar = BigUint::zero();

        let mut steps = BigUint::zero();
        while steps < max_steps {
            curr_point = curr_point.double(p);
            curr_scalar += BigUint::one();
            steps += BigUint::one();

            if &curr_scalar >= n {
                return Some(curr_scalar);
            }
        }

        None
    }
}
