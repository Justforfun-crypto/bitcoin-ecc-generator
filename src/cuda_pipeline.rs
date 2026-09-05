use std::sync::mpsc::{channel, Receiver, Sender};
use num_bigint::BigUint;
use num_traits::Num;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointJacobian {
    pub x: [u64; 4],
    pub y: [u64; 4],
    pub z: [u64; 4],
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: u64,
    pub input_a: Vec<u64>,
    pub input_b: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct WorkResult {
    pub id: u64,
    pub output: Vec<u64>,
}

pub struct GpuPipelineManager {
    num_streams: usize,
    device_id: i32,
    tx: Sender<WorkItem>,
    rx: Receiver<WorkResult>,
}

fn limbs_to_biguint(limbs: &[u64]) -> BigUint {
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().take(4).enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

fn biguint_to_limbs(val: &BigUint) -> Vec<u64> {
    let mut bytes = val.to_bytes_le();
    bytes.resize(32, 0);
    let mut limbs = vec![0u64; 4];
    for i in 0..4 {
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        limbs[i] = u64::from_le_bytes(chunk);
    }
    limbs
}

impl GpuPipelineManager {
    pub fn new(num_streams: usize, device_id: i32) -> Self {
        let (tx_item, rx_item) = channel::<WorkItem>();
        let (tx_res, rx_res) = channel::<WorkResult>();

        let p = BigUint::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap();
        let r = BigUint::from(1u32) << 256usize;
        let r_inv = r.modinv(&p).unwrap();

        std::thread::spawn(move || {
            while let Ok(item) = rx_item.recv() {
                let a = limbs_to_biguint(&item.input_a);
                let b = limbs_to_biguint(&item.input_b);

                let prod = &a * &b;
                let mont_prod = (&prod * &r_inv) % &p;

                let output = biguint_to_limbs(&mont_prod);

                let _ = tx_res.send(WorkResult {
                    id: item.id,
                    output,
                });
            }
        });

        Self {
            num_streams,
            device_id,
            tx: tx_item,
            rx: rx_res,
        }
    }

    pub fn submit(&self, item: WorkItem) -> Result<(), String> {
        self.tx.send(item).map_err(|e| e.to_string())
    }

    pub fn collect_blocking(&self) -> Result<WorkResult, String> {
        self.rx.recv().map_err(|e| e.to_string())
    }
}
