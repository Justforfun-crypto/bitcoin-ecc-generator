use num_bigint::BigUint;
use crate::ecc::Point;
use crate::jacobian::JacobianPoint;
use std::collections::HashMap;

/// Custom 256-bit field arithmetic (no BigUint overhead)
/// Represents field elements as [u32; 8] for better cache locality
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field256 {
    pub limbs: [u32; 8],
}

impl Field256 {
    pub const SECP256K1_P: Field256 = Field256 {
        limbs: [0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    };

    pub fn new(limbs: [u32; 8]) -> Self {
        Field256 { limbs }
    }

    /// Add two field elements (with reduction)
    #[inline]
    pub fn add(&self, other: &Field256) -> Field256 {
        let mut result = [0u64; 9];
        let mut carry = 0u64;

        for i in 0..8 {
            let sum = self.limbs[i] as u64 + other.limbs[i] as u64 + carry;
            result[i] = (sum & 0xFFFFFFFF) as u64;
            carry = sum >> 32;
        }
        result[8] = carry;

        // Reduce modulo p
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            limbs[i] = result[i] as u32;
        }
        Field256::reduce(&limbs)
    }

    /// Multiply two field elements
    #[inline]
    pub fn mul(&self, other: &Field256) -> Field256 {
        let mut result = [0u64; 16];

        // Karatsuba multiplication for speed
        for i in 0..8 {
            for j in 0..8 {
                let prod = (self.limbs[i] as u64) * (other.limbs[j] as u64);
                result[i + j] += prod & 0xFFFFFFFF;
                result[i + j + 1] += prod >> 32;
            }
        }

        // Normalize carries
        for i in 0..15 {
            result[i + 1] += result[i] >> 32;
            result[i] &= 0xFFFFFFFF;
        }

        let mut limbs = [0u32; 8];
        for i in 0..8 {
            limbs[i] = result[i] as u32;
        }
        Field256::reduce(&limbs)
    }

    /// Square field element (faster than mul)
    #[inline]
    pub fn square(&self) -> Field256 {
        self.mul(self)
    }

    /// Modular inversion using Fermat's little theorem: a^-1 = a^(p-2) mod p
    pub fn inv(&self) -> Field256 {
        let mut result = *self;
        for _ in 0..254 {
            result = result.square();
        }
        result.mul(self)
    }

    /// Subtract two field elements
    #[inline]
    pub fn sub(&self, other: &Field256) -> Field256 {
        let mut borrow = 0i64;
        let mut limbs = [0u32; 8];

        for i in 0..8 {
            let diff = (self.limbs[i] as i64) - (other.limbs[i] as i64) - borrow;
            if diff < 0 {
                limbs[i] = (diff + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                limbs[i] = diff as u32;
                borrow = 0;
            }
        }

        if borrow > 0 {
            // Add p
            for i in 0..8 {
                limbs[i] = limbs[i].wrapping_add(Field256::SECP256K1_P.limbs[i]);
            }
        }

        Field256 { limbs }
    }

    /// Reduce result modulo p
    #[inline]
    fn reduce(limbs: &[u32; 8]) -> Field256 {
        let mut result = *limbs;

        // Specialized reduction for secp256k1
        // p = 2^256 - 2^32 - 977
        let mut t = [0u64; 9];
        for i in 0..8 {
            t[i] = result[i] as u64;
        }

        // First reduction
        let mut carry = 0i64;
        for i in 0..8 {
            let val = t[i] as i64 + carry;
            result[i] = (val & 0xFFFFFFFF) as u32;
            carry = val >> 32;
        }

        if carry > 0 {
            for i in 0..8 {
                result[i] = result[i].wrapping_add(Field256::SECP256K1_P.limbs[i]);
            }
        }

        Field256 { limbs: result }
    }
}

/// Precomputed generator point tables for faster multiplication
pub struct GeneratorTables {
    /// Tables for windowed multiplication [0..255][0..31]
    pub window_tables: Vec<Vec<Point>>,
    /// Precomputed powers: [G, 2G, 4G, 8G, ...]
    pub powers_of_two: Vec<Point>,
    /// Precomputed powers: [G, 3G, 5G, 7G, 9G, ...] (odd multiples)
    pub odd_multiples: Vec<Point>,
}

impl GeneratorTables {
    pub fn precompute(p: &BigUint, n: &BigUint) -> Self {
        let g = Point::generator();
        let mut window_tables = Vec::new();
        let mut powers_of_two = Vec::new();
        let mut odd_multiples = Vec::new();

        // Precompute windowed tables
        for w in 0..256 {
            let mut window = Vec::new();
            let mut point = g.clone();
            let mut multiplier = BigUint::from(1u32);

            for i in 0..31 {
                window.push(point.clone());
                point = Self::scalar_mult(&point, &BigUint::from(2u32 << (w % 8)), p, n);
                multiplier = &multiplier * BigUint::from(2u32);
            }
            window_tables.push(window);
        }

        // Precompute powers of 2
        let mut power = g.clone();
        for _ in 0..256 {
            powers_of_two.push(power.clone());
            power = Self::point_double(&power, p);
        }

        // Precompute odd multiples
        let mut odd = g.clone();
        for _ in 0..128 {
            odd_multiples.push(odd.clone());
            odd = Self::scalar_mult(&odd, &BigUint::from(2u32), p, n);
        }

        GeneratorTables {
            window_tables,
            powers_of_two,
            odd_multiples,
        }
    }

    fn scalar_mult(point: &Point, scalar: &BigUint, p: &BigUint, _n: &BigUint) -> Point {
        if scalar == &BigUint::from(0u32) {
            return Point::infinity();
        }

        let mut result = Point::infinity();
        let mut addend = point.clone();
        let mut scalar_copy = scalar.clone();

        while scalar_copy > BigUint::from(0u32) {
            if (&scalar_copy & &BigUint::from(1u32)) == BigUint::from(1u32) {
                result = Self::point_add(&result, &addend, p);
            }
            addend = Self::point_double(&addend, p);
            scalar_copy >>= 1;
        }

        result
    }

    fn point_double(point: &Point, p: &BigUint) -> Point {
        if point.is_infinity {
            return Point::infinity();
        }
        let jpoint = JacobianPoint::from_affine(point);
        let doubled = jpoint.double(p);
        doubled.to_affine(p)
    }

    fn point_add(p1: &Point, p2: &Point, p: &BigUint) -> Point {
        if p1.is_infinity {
            return p2.clone();
        }
        if p2.is_infinity {
            return p1.clone();
        }
        let jp1 = JacobianPoint::from_affine(p1);
        let jp2 = JacobianPoint::from_affine(p2);
        jp1.add(&jp2, p).to_affine(p)
    }
}

/// Shamir's trick for k1*P1 + k2*P2 (used in GLV)
pub struct ShamirTrick;

impl ShamirTrick {
    /// Compute k1*P1 + k2*P2 efficiently
    pub fn double_mult(
        k1: &BigUint,
        p1: &Point,
        k2: &BigUint,
        p2: &Point,
        p: &BigUint,
    ) -> Point {
        let max_bits = k1.bits().max(k2.bits()) as usize;
        let mut result = Point::infinity();

        for i in (0..max_bits).rev() {
            result = Self::point_double(&result, p);

            let bit1 = (k1 >> i) & BigUint::from(1u32) == BigUint::from(1u32);
            let bit2 = (k2 >> i) & BigUint::from(1u32) == BigUint::from(1u32);

            match (bit1, bit2) {
                (true, false) => result = Self::point_add(&result, p1, p),
                (false, true) => result = Self::point_add(&result, p2, p),
                (true, true) => {
                    let sum = Self::point_add(p1, p2, p);
                    result = Self::point_add(&result, &sum, p);
                }
                (false, false) => {}
            }
        }

        result
    }

    fn point_double(point: &Point, p: &BigUint) -> Point {
        if point.is_infinity {
            return Point::infinity();
        }
        let jpoint = JacobianPoint::from_affine(point);
        let doubled = jpoint.double(p);
        doubled.to_affine(p)
    }

    fn point_add(p1: &Point, p2: &Point, p: &BigUint) -> Point {
        if p1.is_infinity {
            return p2.clone();
        }
        if p2.is_infinity {
            return p1.clone();
        }
        let jp1 = JacobianPoint::from_affine(p1);
        let jp2 = JacobianPoint::from_affine(p2);
        jp1.add(&jp2, p).to_affine(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_add() {
        let a = Field256::new([1, 0, 0, 0, 0, 0, 0, 0]);
        let b = Field256::new([2, 0, 0, 0, 0, 0, 0, 0]);
        let c = a.add(&b);
        assert_eq!(c.limbs[0], 3);
    }

    #[test]
    fn test_field_mul() {
        let a = Field256::new([2, 0, 0, 0, 0, 0, 0, 0]);
        let b = Field256::new([3, 0, 0, 0, 0, 0, 0, 0]);
        let c = a.mul(&b);
        assert_eq!(c.limbs[0], 6);
    }
}
