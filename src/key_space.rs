use num_bigint::BigUint;
use num_traits::ToPrimitive;

pub struct KeySpaceReducer {
    pub range_start: BigUint,
    pub range_end: BigUint,
}

impl KeySpaceReducer {
    pub fn new(range_start: BigUint, range_end: BigUint) -> Self {
        Self {
            range_start,
            range_end,
        }
    }

    pub fn narrow_range(&mut self, factor: f64) {
        let diff = &self.range_end - &self.range_start;
        let diff_float = diff.to_f64().unwrap_or(0.0);
        let new_range_len = BigUint::from((diff_float * factor) as u64);
        self.range_end = &self.range_start + new_range_len;
    }
}
