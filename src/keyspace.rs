use num_bigint::BigUint;

pub struct KeyspaceFilter {
    pub start: BigUint,
    pub end: BigUint,
    pub stride: u64,
}

impl KeyspaceFilter {
    pub fn new(start: impl Into<BigUint>, end: impl Into<BigUint>, stride: u64) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            stride,
        }
    }

    pub fn chunk_ranges(&self, chunk_size: u64) -> Vec<(BigUint, BigUint)> {
        let mut chunks = vec![];
        let mut curr = self.start.clone();
        let step = BigUint::from(chunk_size);
        let end = &self.end;

        while curr < *end {
            let next = std::cmp::min(&curr + &step, end.clone());
            chunks.push((curr.clone(), next.clone()));
            curr = next;
        }
        chunks
    }
}
