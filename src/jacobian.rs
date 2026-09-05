use num_bigint::BigUint;
use num_traits::Zero;
use crate::ecc::Point;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianPoint {
    pub x: BigUint,
    pub y: BigUint,
    pub z: BigUint,
}

impl JacobianPoint {
    pub fn new(x: BigUint, y: BigUint, z: BigUint) -> Self {
        Self { x, y, z }
    }

    pub fn from_affine(point: &Point) -> Self {
        if point.is_infinity {
            Self::new(BigUint::zero(), BigUint::zero(), BigUint::zero())
        } else {
            Self::new(
                point.x.clone(),
                point.y.clone(),
                BigUint::from(1u32),
            )
        }
    }

    pub fn to_affine(&self, p: &BigUint) -> Point {
        if self.z.is_zero() {
            return Point::infinity();
        }

        let z_inv = match Self::mod_inverse(&self.z, p) {
            Some(inv) => inv,
            None => return Point::infinity(),
        };

        let z_inv_sq = (&z_inv * &z_inv) % p;
        let z_inv_cube = (&z_inv_sq * &z_inv) % p;

        let x = (&self.x * &z_inv_sq) % p;
        let y = (&self.y * &z_inv_cube) % p;

        Point::new(x, y)
    }

    pub fn double(&self, p: &BigUint) -> Self {
        if self.y.is_zero() || self.z.is_zero() {
            return Self::new(BigUint::zero(), BigUint::zero(), BigUint::zero());
        }

        let xx = (&self.x * &self.x) % p;
        let yy = (&self.y * &self.y) % p;
        let yyyy = (&yy * &yy) % p;
        let s = (BigUint::from(2u32) * ((&self.x + &yy) * (&self.x + &yy) - &xx - &yyyy)) % p;
        let m = (BigUint::from(3u32) * &xx) % p;

        let t = (&m * &m) % p;
        let x3 = (p + &t - (BigUint::from(2u32) * &s) % p) % p;
        let y3 = (p + (&m * (&s + p - &x3)) % p - (BigUint::from(8u32) * &yyyy) % p) % p;
        let z3 = (BigUint::from(2u32) * &self.y * &self.z) % p;

        Self::new(x3, y3, z3)
    }

    pub fn add(&self, other: &Self, p: &BigUint) -> Self {
        if self.z.is_zero() {
            return other.clone();
        }
        if other.z.is_zero() {
            return self.clone();
        }

        let z1_sq = (&self.z * &self.z) % p;
        let z2_sq = (&other.z * &other.z) % p;

        let u1 = (&self.x * &z2_sq) % p;
        let u2 = (&other.x * &z1_sq) % p;

        let s1 = (&self.y * &other.z * &z2_sq) % p;
        let s2 = (&other.y * &self.z * &z1_sq) % p;

        if u1 == u2 {
            if s1 != s2 {
                return Self::new(BigUint::zero(), BigUint::zero(), BigUint::zero());
            } else {
                return self.double(p);
            }
        }

        let h = (p + &u2 - &u1) % p;
        let r = (p + &s2 - &s1) % p;
        let h_sq = (&h * &h) % p;
        let h_cube = (&h * &h_sq) % p;

        let x3 = (p + (p + (&r * &r) % p - &h_cube) % p - (BigUint::from(2u32) * &u1 * &h_sq) % p) % p;
        let y3 = (p + (&r * ((&u1 * &h_sq) % p + p - &x3)) % p - (&s1 * &h_cube) % p) % p;
        let z3 = (&self.z * &other.z * &h) % p;

        Self::new(x3, y3, z3)
    }

    fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
        let mut mn = (m.clone(), a.clone());
        let mut xy = (BigUint::zero(), BigUint::from(1u32));
        let mut sign = false;

        while !mn.1.is_zero() {
            let q = &mn.0 / &mn.1;
            let r = &mn.0 % &mn.1;
            mn = (mn.1, r);

            let t = &q * &xy.1 + &xy.0;
            xy = (xy.1, t);
            sign = !sign;
        }

        if mn.0 > BigUint::from(1u32) {
            return None;
        }

        if sign {
            Some(m - xy.0)
        } else {
            Some(xy.0)
        }
    }
}
