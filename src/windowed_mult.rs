use num_bigint::BigUint;
use crate::ecc::Point;
use crate::jacobian::JacobianPoint;

/// Windowed scalar multiplication for faster computation
/// Pre-computes [1*G, 2*G, 3*G, ..., (2^w-1)*G] and uses w-bit windows
pub struct WindowedMultiplier {
    /// Window width (typically 4-8)
    pub window_width: usize,
    /// Pre-computed table: window_table[i][j] = ((i+1) * 2^(j*w)) * G
    pub window_table: Vec<Vec<Point>>,
}

impl WindowedMultiplier {
    pub fn new(window_width: usize, p: &BigUint, n: &BigUint) -> Self {
        let g = Point::generator();
        let table_size = 1 << window_width; // 2^w
        let mut window_table = Vec::new();

        // Pre-compute [0*G, 1*G, 2*G, ..., (2^w-1)*G]
        let mut base = Point::infinity();
        for i in 0..table_size {
            window_table.push(base.clone());
            base = scalar_mult(&base, &BigUint::from(1u32), p, n);
        }

        WindowedMultiplier {
            window_width,
            window_table,
        }
    }

    /// Scalar multiplication using windowed method
    pub fn multiply(&self, scalar: &BigUint, p: &BigUint) -> Point {
        let window_mask = (1u32 << self.window_width) - 1;
        let mut result = Point::infinity();
        let scalar_bits = scalar.to_bytes_le();
        let bit_length = (scalar_bits.len() * 8) as i32;

        // Process from MSB to LSB
        let mut i = bit_length - 1;
        while i >= 0 {
            // Square for window width
            for _ in 0..self.window_width {
                result = self.point_double(&result, p);
                i -= 1;
                if i < 0 {
                    break;
                }
            }

            // Extract window
            let mut window_value = 0u32;
            for j in 0..self.window_width {
                let bit_index = i - j as i32;
                if bit_index >= 0 {
                    let byte_idx = (bit_index / 8) as usize;
                    let bit_in_byte = (bit_index % 8) as usize;
                    if byte_idx < scalar_bits.len() {
                        let bit = (scalar_bits[byte_idx] >> bit_in_byte) & 1;
                        window_value = (window_value << 1) | (bit as u32);
                    }
                }
            }

            // Add pre-computed point
            if window_value > 0 && window_value < self.window_table.len() as u32 {
                result = self.point_add(&result, &self.window_table[window_value as usize], p);
            }
        }

        result
    }

    fn point_double(&self, point: &Point, p: &BigUint) -> Point {
        if point.is_infinity {
            return Point::infinity();
        }
        let jpoint = JacobianPoint::from_affine(point);
        let doubled = jpoint.double(p);
        doubled.to_affine(p)
    }

    fn point_add(&self, p1: &Point, p2: &Point, p: &BigUint) -> Point {
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

/// Cached windowed multiplication with pre-computed values
pub struct CachedWindowedMult {
    multiplier: WindowedMultiplier,
    cache: std::collections::HashMap<String, Point>,
}

impl CachedWindowedMult {
    pub fn new(window_width: usize, p: &BigUint, n: &BigUint) -> Self {
        CachedWindowedMult {
            multiplier: WindowedMultiplier::new(window_width, p, n),
            cache: std::collections::HashMap::new(),
        }
    }

    pub fn multiply(&mut self, scalar: &BigUint, p: &BigUint) -> Point {
        let key = format!("{}:{}:{}", scalar.to_str_radix(16), p.to_str_radix(16), self.multiplier.window_width);
        
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let result = self.multiplier.multiply(scalar, p);
        self.cache.insert(key, result.clone());
        result
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
            result = point_add(&result, &addend, p);
        }
        addend = point_double(&addend, p);
        scalar_copy >>= 1;
    }

    result
}

fn point_double(point: &Point, p: &BigUint) -> Point {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_windowed_mult() {
        let p = BigUint::from_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F").unwrap();
        let n = BigUint::from_str("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141").unwrap();
        let multiplier = WindowedMultiplier::new(4, &p, &n);
        let scalar = BigUint::from(100u32);
        let result = multiplier.multiply(&scalar, &p);
        assert!(!result.is_infinity);
    }
}
