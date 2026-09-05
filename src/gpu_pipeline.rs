use crate::gpu_orchestration::{GpuOrchestrator, PinnedBuffer, CudaStreamWrapper};
use cudarc::driver::CudaError;

pub struct DoubleBufferPipeline<T: Copy + Default> {
    pub stream_a: CudaStreamWrapper,
    pub stream_b: CudaStreamWrapper,
    pub host_buf_a: PinnedBuffer<T>,
    pub host_buf_b: PinnedBuffer<T>,
}

impl GpuOrchestrator {
    /// Executes a batched workload using 2-stream double buffering 
    /// to overlap PCIe DMA host transfers with GPU execution.
    pub fn process_batched_pipeline<T: Copy + Default>(
        &self,
        chunks: &[Vec<T>],
        chunk_size: usize,
    ) -> Result<(), CudaError> {
        let stream_a = self.create_stream()?;
        let stream_b = self.create_stream()?;

        let mut host_a = self.allocate_pinned::<T>(chunk_size)?;
        let mut host_b = self.allocate_pinned::<T>(chunk_size)?;

        let mut dev_a = self.device.alloc_zeros::<T>(chunk_size)?;
        let mut dev_b = self.device.alloc_zeros::<T>(chunk_size)?;

        for (i, chunk) in chunks.chunks(2).enumerate() {
            // --- STREAM A: Process Chunk 2i ---
            if let Some(data_a) = chunk.get(0) {
                host_a.as_mut_slice()[..data_a.len()].copy_from_slice(data_a);

                // 1. Queue async H2D copy on Stream A
                self.copy_pinned_to_device_async(&host_a, &mut dev_a, &stream_a)?;

                // 2. Launch CUDA kernel on Stream A (Non-blocking)
                // self.launch_kernel_async(&dev_a, &stream_a)?;

                // 3. Queue async D2H copy back on Stream A
                self.copy_device_to_pinned_async(&dev_a, &mut host_a, &stream_a)?;
            }

            // --- STREAM B: Process Chunk 2i + 1 (Concurrently) ---
            if let Some(data_b) = chunk.get(1) {
                host_b.as_mut_slice()[..data_b.len()].copy_from_slice(data_b);

                // 1. Queue async H2D copy on Stream B while Stream A computes
                self.copy_pinned_to_device_async(&host_b, &mut dev_b, &stream_b)?;

                // 2. Launch CUDA kernel on Stream B
                // self.launch_kernel_async(&dev_b, &stream_b)?;

                // 3. Queue async D2H copy back on Stream B
                self.copy_device_to_pinned_async(&dev_b, &mut host_b, &stream_b)?;
            }

            // --- SYNCHRONIZATION POINT ---
            // Wait ONLY for Stream A's batch to finish so the host can process Chunk 2i
            stream_a.synchronize()?;
            // Process host_a results here...

            // Wait for Stream B's batch to finish so the host can process Chunk 2i + 1
            stream_b.synchronize()?;
            // Process host_b results here...
        }

        Ok(())
    }
}
