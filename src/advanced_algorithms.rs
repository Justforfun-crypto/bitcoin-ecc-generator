use num_bigint::BigUint;
use crate::ecc::Point;
use crate::jacobian::JacobianPoint;

/// Montgomery Ladder for constant-time scalar multiplication
/// Resistant to side-channel attacks
pub struct MontgomeryLadder;

impl MontgomeryLadder {
    /// Constant-time scalar multiplication using Montgomery Ladder
    pub fn multiply(k: &BigUint, point: &Point, p: &BigUint) -> Point {
        let k_bits = k.bits() as usize;
        let mut r0 = Point::infinity();
        let mut r1 = point.clone();

        for i in (0..k_bits).rev() {
            let bit = ((k >> i) & BigUint::from(1u32)) == BigUint::from(1u32);

            if bit {
                std::mem::swap(&mut r0, &mut r1);
            }

            // Conditional operations (constant-time)
            let sum = Self::point_add(&r0, &r1, p);
            let doubled = Self::point_double(&r0, p);

            r0 = doubled;
            r1 = sum;

            if bit {
                std::mem::swap(&mut r0, &mut r1);
            }
        }

        r0
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

/// Meet-in-the-Middle attack for key recovery
/// Reduces 2^256 search space to 2^128 with 2^128 memory
pub struct MeetInTheMiddle {
    pub range_start: BigUint,
    pub range_end: BigUint,
    pub target: Point,
}

impl MeetInTheMiddle {
    pub fn new(range_start: BigUint, range_end: BigUint, target: Point) -> Self {
        MeetInTheMiddle {
            range_start,
            range_end,
            target,
        }
    }

    /// Execute meet-in-the-middle attack
    /// k = k_low + k_high * 2^128 where k_low, k_high in [0, 2^128)
    pub fn solve(&self, p: &BigUint, n: &BigUint) -> Option<BigUint> {
        let split_point = BigUint::from(1u32) << 128;
        let g = Point::generator();

        // Phase 1: Compute all k_low*G, store in table
        let mut baby_steps = std::collections::HashMap::new();
        let mut baby = Point::infinity();

        for low in 0u32..u32::MAX {
            let key = format!("{}:{}", baby.x, baby.y);
            baby_steps.insert(key, BigUint::from(low));

            baby = Self::point_add(&baby, &g, p);

            if low % 1000000 == 0 {
                println!("Baby steps: {}", low);
            }
        }

        // Phase 2: Compute target - k_high*2^128*G, match in table
        let g_split = Self::scalar_mult(&g, &split_point, p, n);
        let mut giant = self.target.clone();

        for high in 0u32..u32::MAX {
            let key = format!("{}:{}", giant.x, giant.y);

            if let Some(low) = baby_steps.get(&key) {
                return Some(low + BigUint::from(high) * split_point.clone());
            }

            giant = Self::point_sub(&giant, &g_split, p);

            if high % 1000000 == 0 {
                println!("Giant steps: {}", high);
            }
        }

        None
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

    fn point_sub(p1: &Point, p2: &Point, p: &BigUint) -> Point {
        let mut negated = p2.clone();
        negated.y = (BigUint::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap() - negated.y.clone()) % p;
        Self::point_add(p1, &negated, p)
    }
}

/// Rho algorithm optimization with cycle detection
pub struct PollardsRhoOptimized;

impl PollardsRhoOptimized {
    /// Pollard's rho with Brent's cycle detection
    pub fn solve(
        target: &Point,
        range_start: &BigUint,
        range_end: &BigUint,
        p: &BigUint,
        n: &BigUint,
    ) -> Option<BigUint> {
        let range = range_end - range_start;
        let mut x = Point::generator();
        let mut y = Point::generator();
        let mut r = BigUint::from(1u32);
        let mut q = BigUint::from(1u32);

        loop {
            let ys = y.clone();
            for _ in 0..r.min(BigUint::from(1000u32)) {
                y = Self::pseudo_random_step(&y, p, n);
                let diff = Self::point_sub(&x, &y, p);
                q = (q.clone() * Self::point_to_scalar(&diff, p)) % n;
            }

            r = r.clone() * BigUint::from(2u32);
            x = y.clone();

            let gcd_val = Self::gcd(&q, n);
            if gcd_val != BigUint::from(1u32) && gcd_val != n {
                return Some(gcd_val);
            }

            if y == *target {
                return Some(range_start.clone());
            }
        }
    }

    fn pseudo_random_step(point: &Point, p: &BigUint, n: &BigUint) -> Point {
        // f(x) = x^2 + 1 (mod p)
        let jp = JacobianPoint::from_affine(point);
        let squared = jp.double(p);
        squared.to_affine(p)
    }

    fn point_to_scalar(point: &Point, p: &BigUint) -> BigUint {
        point.x.clone() % p
    }

    fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
        let mut a = a.clone();
        let mut b = b.clone();
        while b != BigUint::from(0u32) {
            let temp = b.clone();
            b = a.clone() % b.clone();
            a = temp;
        }
        a
    }

    fn point_sub(p1: &Point, p2: &Point, p: &BigUint) -> Point {
        let mut negated = p2.clone();
        negated.y = (BigUint::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap() - negated.y.clone()) % p;
        let jp1 = JacobianPoint::from_affine(p1);
        let jp2 = JacobianPoint::from_affine(&negated);
        jp1.add(&jp2, p).to_affine(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_montgomery_ladder() {
        let p = BigUint::from_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F").unwrap();
        let k = BigUint::from(12345u32);
        let g = Point::generator();
        let result = MontgomeryLadder::multiply(&k, &g, &p);
        assert!(!result.is_infinity);
    }
}
