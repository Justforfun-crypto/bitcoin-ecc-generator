pub struct KeyspaceFilter {
    start: u128,
    end: u128,
    step: u128,
}

impl KeyspaceFilter {
    pub fn new(start: u128, end: u128, step: u128) -> Self {
        Self { start, end, step }
    }

    pub fn contains(&self, key: u128) -> bool {
        key >= self.start && key <= self.end && (key - self.start) % self.step == 0
    }
}
