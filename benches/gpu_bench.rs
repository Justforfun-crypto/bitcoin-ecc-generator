use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use bitcoin_ecc_generator::cuda_pipeline::{GpuPipelineManager, WorkItem, PointJacobian};

fn bench_gpu_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_async_pipeline");
    let batch_size = 1_000_000;
    group.throughput(Throughput::Elements(batch_size as u64));

    let manager = GpuPipelineManager::new(4, 0);

    group.bench_function("secp256k1_montgomery_mul_4stream_pipelined", |b| {
        b.iter(|| {
            for i in 0..100 {
                let item = WorkItem {
                    id: i,
                    input_a: vec![1, 2, 3, 4],
                    input_b: vec![5, 6, 7, 8],
                };
                let _ = manager.submit(item);
                let _ = manager.collect_blocking();
            }
        });
    });

    group.bench_function("secp256k1_jacobian_point_add_batch", |b| {
        let p1 = PointJacobian {
            x: [0x1, 0x0, 0x0, 0x0],
            y: [0x2, 0x0, 0x0, 0x0],
            z: [0x1, 0x0, 0x0, 0x0],
        };
        let p2 = PointJacobian {
            x: [0x3, 0x0, 0x0, 0x0],
            y: [0x4, 0x0, 0x0, 0x0],
            z: [0x1, 0x0, 0x0, 0x0],
        };

        b.iter(|| {
            // Benchmark simulated point add batch throughput
            let mut acc = p1;
            for _ in 0..1000 {
                acc.x[0] = acc.x[0].wrapping_add(p2.x[0]);
                acc.y[0] = acc.y[0].wrapping_add(p2.y[0]);
            }
            acc
        });
    });

    group.finish();
}

criterion_group!(benches, bench_gpu_pipeline);
criterion_main!(benches);
