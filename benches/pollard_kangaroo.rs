use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use bitcoin_ecc_generator::gpu_orchestration::GpuOrchestrator;
use bitcoin_ecc_generator::cuda_gpu::CudaDevice;
use cuda_runtime_sys::{self as sys, cudaMemcpyKind};
use std::sync::Arc;

pub fn bench_transfer_pipeline_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_transfer_pipeline");

    // Parameter sweep across 2^20 (1M), 2^21 (2M), and 2^22 (4M) elements
    let element_counts = vec![1_048_576, 2_097_152, 4_194_304];
    let bytes_per_elem = std::mem::size_of::<u64>() * 4;

    let device = Arc::new(CudaDevice::new(0).expect("Failed to initialize CUDA device"));
    let orchestrator = GpuOrchestrator::new(device.clone());

    for &num_elements in &element_counts {
        let total_bytes = (num_elements * bytes_per_elem) as u64;
        group.throughput(Throughput::Bytes(total_bytes));

        // 1. Single Stream Unpinned Sync Baseline
        group.bench_function(
            BenchmarkId::new("single_stream_unpinned_sync", num_elements),
            |b| {
                let mut host_data = vec![0u8; total_bytes as usize];
                let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

                unsafe {
                    sys::cudaMalloc(&mut dev_ptr as *mut _ as *mut _, total_bytes as usize);
                }

                b.iter(|| unsafe {
                    sys::cudaMemcpy(
                        dev_ptr,
                        host_data.as_ptr() as *const _,
                        total_bytes as usize,
                        cudaMemcpyKind::cudaMemcpyHostToDevice,
                    );
                    sys::cudaMemcpy(
                        host_data.as_mut_ptr() as *mut _,
                        dev_ptr as *const _,
                        total_bytes as usize,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    );
                });

                unsafe {
                    sys::cudaFree(dev_ptr);
                }
            },
        );

        // 2. Double-Stream Pinned Async
        group.bench_function(
            BenchmarkId::new("two_stream_pinned_async", num_elements),
            |b| {
                let chunk_bytes = (total_bytes as usize) / 2;

                let mut pinned_a = orchestrator.allocate_pinned::<u8>(chunk_bytes).unwrap();
                let mut pinned_b = orchestrator.allocate_pinned::<u8>(chunk_bytes).unwrap();

                let mut dev_a: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut dev_b: *mut std::ffi::c_void = std::ptr::null_mut();

                unsafe {
                    sys::cudaMalloc(&mut dev_a as *mut _ as *mut _, chunk_bytes);
                    sys::cudaMalloc(&mut dev_b as *mut _ as *mut _, chunk_bytes);
                }

                let stream_a = orchestrator.create_stream().unwrap();
                let stream_b = orchestrator.create_stream().unwrap();

                b.iter(|| {
                    orchestrator
                        .copy_pinned_to_device_async(&pinned_a, dev_a, &stream_a)
                        .unwrap();
                    orchestrator
                        .copy_device_to_pinned_async(dev_a, &mut pinned_a, &stream_a)
                        .unwrap();

                    orchestrator
                        .copy_pinned_to_device_async(&pinned_b, dev_b, &stream_b)
                        .unwrap();
                    orchestrator
                        .copy_device_to_pinned_async(dev_b, &mut pinned_b, &stream_b)
                        .unwrap();

                    stream_a.synchronize().unwrap();
                    stream_b.synchronize().unwrap();
                });

                unsafe {
                    sys::cudaFree(dev_a);
                    sys::cudaFree(dev_b);
                }
            },
        );

        // 3. Triple-Buffering / 3-Stream Pinned Async
        group.bench_function(
            BenchmarkId::new("three_stream_pinned_async", num_elements),
            |b| {
                let chunk_bytes = (total_bytes as usize) / 3;

                let mut pinned_a = orchestrator.allocate_pinned::<u8>(chunk_bytes).unwrap();
                let mut pinned_b = orchestrator.allocate_pinned::<u8>(chunk_bytes).unwrap();
                let mut pinned_c = orchestrator.allocate_pinned::<u8>(chunk_bytes).unwrap();

                let mut dev_a: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut dev_b: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut dev_c: *mut std::ffi::c_void = std::ptr::null_mut();

                unsafe {
                    sys::cudaMalloc(&mut dev_a as *mut _ as *mut _, chunk_bytes);
                    sys::cudaMalloc(&mut dev_b as *mut _ as *mut _, chunk_bytes);
                    sys::cudaMalloc(&mut dev_c as *mut _ as *mut _, chunk_bytes);
                }

                let stream_a = orchestrator.create_stream().unwrap();
                let stream_b = orchestrator.create_stream().unwrap();
                let stream_c = orchestrator.create_stream().unwrap();

                b.iter(|| {
                    orchestrator
                        .copy_pinned_to_device_async(&pinned_a, dev_a, &stream_a)
                        .unwrap();
                    orchestrator
                        .copy_device_to_pinned_async(dev_a, &mut pinned_a, &stream_a)
                        .unwrap();

                    orchestrator
                        .copy_pinned_to_device_async(&pinned_b, dev_b, &stream_b)
                        .unwrap();
                    orchestrator
                        .copy_device_to_pinned_async(dev_b, &mut pinned_b, &stream_b)
                        .unwrap();

                    orchestrator
                        .copy_pinned_to_device_async(&pinned_c, dev_c, &stream_c)
                        .unwrap();
                    orchestrator
                        .copy_device_to_pinned_async(dev_c, &mut pinned_c, &stream_c)
                        .unwrap();

                    stream_a.synchronize().unwrap();
                    stream_b.synchronize().unwrap();
                    stream_c.synchronize().unwrap();
                });

                unsafe {
                    sys::cudaFree(dev_a);
                    sys::cudaFree(dev_b);
                    sys::cudaFree(dev_c);
                }
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_transfer_pipeline_comparison);
criterion_main!(benches);
