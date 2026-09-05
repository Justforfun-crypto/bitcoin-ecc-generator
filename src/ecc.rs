use num_bigint::BigUint;
use num_traits::Num;
use num_traits::Zero;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    pub x: BigUint,
    pub y: BigUint,
    pub is_infinity: bool,
}

impl Point {
    pub fn new(x: BigUint, y: BigUint) -> Self {
        Self {
            x,
            y,
            is_infinity: false,
        }
    }

    pub fn infinity() -> Self {
        Self {
            x: BigUint::zero(),
            y: BigUint::zero(),
            is_infinity: true,
        }
    }

    pub fn generator() -> Self {
        let gx = BigUint::from_str_radix("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16).unwrap();
        let gy = BigUint::from_str_radix("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16).unwrap();
        Self::new(gx, gy)
    }
}
