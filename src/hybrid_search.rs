use num_bigint::BigUint;
use crate::ecc::Point;
use crate::pollard_kangaroo::PollardKangaroo;
use crate::key_space::KeySpaceReducer;
use rayon::prelude::*;

/// Hybrid search combining multiple algorithms for optimal performance
pub struct HybridSearch {
    pub target: Point,
    pub range_start: BigUint,
    pub range_end: BigUint,
}

impl HybridSearch {
    pub fn new(target: Point, range_start: BigUint, range_end: BigUint) -> Self {
        HybridSearch {
            target,
            range_start,
            range_end,
        }
    }

    /// Smart algorithm selection based on range size
    pub fn get_best_algorithm(&self) -> Algorithm {
        let range_size = &self.range_end - &self.range_start;
        let bits = range_size.bits();

        if bits < 20 {
            Algorithm::BruteForce
        } else if bits < 40 {
            Algorithm::PollardRho
        } else if bits < 60 {
            Algorithm::PollardKangaroo
        } else {
            Algorithm::HybridParallel
        }
    }

    /// Parallel search across multiple ranges
    pub fn parallel_search(
        &self,
        num_threads: usize,
        p: &BigUint,
        n: &BigUint,
    ) -> Option<BigUint> {
        let ranges = self.divide_ranges(num_threads);

        ranges
            .into_par_iter()
            .find_map_any(|(start, end)| {
                let kangaroo = PollardKangaroo::new(start, end, self.target.clone());
                kangaroo.solve(p, n)
            })
    }

    /// Progressive narrowing search
    pub fn progressive_search(
        &self,
        p: &BigUint,
        n: &BigUint,
    ) -> Option<BigUint> {
        let mut reducer = KeySpaceReducer::new(
            self.range_start.clone(),
            self.range_end.clone(),
        );

        // Round 1: Search full range
        let full_kangaroo = PollardKangaroo::new(
            reducer.range_start.clone(),
            reducer.range_end.clone(),
            self.target.clone(),
        );

        if let Some(result) = full_kangaroo.solve(p, n) {
            return Some(result);
        }

        // Round 2: Narrow range by 50%
        reducer.narrow_range(0.5);
        let narrow_kangaroo = PollardKangaroo::new(
            reducer.range_start.clone(),
            reducer.range_end.clone(),
            self.target.clone(),
        );

        narrow_kangaroo.solve(p, n)
    }

    fn divide_ranges(&self, num_ranges: usize) -> Vec<(BigUint, BigUint)> {
        let mut ranges = Vec::new();
        let range_size = &self.range_end - &self.range_start;
        let chunk_size = range_size / BigUint::from(num_ranges as u64);

        for i in 0..num_ranges {
            let start = self.range_start.clone() + chunk_size.clone() * BigUint::from(i as u64);
            let end = if i == num_ranges - 1 {
                self.range_end.clone()
            } else {
                start.clone() + chunk_size.clone()
            };
            ranges.push((start, end));
        }

        ranges
    }
}

pub enum Algorithm {
    BruteForce,
    PollardRho,
    PollardKangaroo,
    HybridParallel,
}

/// Distributed search coordinator
pub struct DistributedSearch {
    workers: Vec<SearchWorker>,
}

pub struct SearchWorker {
    id: usize,
    range_start: BigUint,
    range_end: BigUint,
}

impl DistributedSearch {
    pub fn new(num_workers: usize, total_range: (BigUint, BigUint)) -> Self {
        let (start, end) = total_range;
        let range_size = end.clone() - start.clone();
        let chunk_size = range_size / BigUint::from(num_workers as u64);

        let mut workers = Vec::new();
        for i in 0..num_workers {
            let worker_start = start.clone() + chunk_size.clone() * BigUint::from(i as u64);
            let worker_end = if i == num_workers - 1 {
                end.clone()
            } else {
                worker_start.clone() + chunk_size.clone()
            };

            workers.push(SearchWorker {
                id: i,
                range_start: worker_start,
                range_end: worker_end,
            });
        }

        DistributedSearch { workers }
    }

    pub fn get_worker_ranges(&self) -> Vec<(usize, BigUint, BigUint)> {
        self.workers
            .iter()
            .map(|w| (w.id, w.range_start.clone(), w.range_end.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_init() {
        let target = Point::generator();
        let start = BigUint::from(1000u32);
        let end = BigUint::from(2000u32);
        let search = HybridSearch::new(target, start, end);
        assert_eq!(search.get_best_algorithm() as u32, Algorithm::BruteForce as u32);
    }

    #[test]
    fn test_distributed_search() {
        let start = BigUint::from(0u32);
        let end = BigUint::from(1000u32);
        let search = DistributedSearch::new(4, (start, end));
        let ranges = search.get_worker_ranges();
        assert_eq!(ranges.len(), 4);
    }
}
