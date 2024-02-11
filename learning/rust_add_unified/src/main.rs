use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DeviceRepr, LaunchAsync};

const N: usize = 10_000_000;
const BENCH: usize = 1000;
const MAX_ERR: f32 = 1e-9;

struct F32Ptr(*const f32);
struct F32MutPtr(*mut f32);

unsafe impl DeviceRepr for F32Ptr {}
unsafe impl DeviceRepr for F32MutPtr {}

fn main() -> anyhow::Result<()> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let ptx = cudarc::nvrtc::compile_ptx(include_str!("../vector_add.cu"))?;
    println!("compiled ptx");

    // load ptx onto device
    dev.load_ptx(ptx, "adder", &["vector_add"])?;
    println!("loaded ptx onto device");

    // using managed memory for iGPU
    let mut a: CUdeviceptr = 0;
    unsafe {
        cudarc::driver::sys::cuMemAllocManaged(
            &mut a as *mut CUdeviceptr,
            std::mem::size_of::<f32>() * N,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST as u32,
        );
    }
    let mut b: CUdeviceptr = 0;
    unsafe {
        cudarc::driver::sys::cuMemAllocManaged(
            &mut b as *mut CUdeviceptr,
            std::mem::size_of::<f32>() * N,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST as u32,
        );
    }
    let mut out: CUdeviceptr = 0;
    unsafe {
        cudarc::driver::sys::cuMemAllocManaged(
            &mut out as *mut CUdeviceptr,
            std::mem::size_of::<f32>() * N,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        );
    }
    println!("allocated managed memory");

    unsafe {
        let view_a = std::slice::from_raw_parts_mut(a as *mut f32, N);
        let view_b = std::slice::from_raw_parts_mut(b as *mut f32, N);
        for i in 0..N {
            view_a[i] = 1.;
            view_b[i] = 2.;
        }
    }
    println!("assigned input values");

    let block_size: u32 = 256;
    let num_blocks: u32 = (N as u32 + block_size - 1) / block_size;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // prefetching hints for performance improvements
    unsafe {
        cudarc::driver::sys::cuStreamAttachMemAsync(
            std::ptr::null_mut(),
            a,
            0,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        )
    };
    unsafe {
        cudarc::driver::sys::cuStreamAttachMemAsync(
            std::ptr::null_mut(),
            b,
            0,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        )
    };

    for _ in 0..BENCH {
        let vector_add = dev.get_func("adder", "vector_add").unwrap();
        unsafe {
            vector_add.launch(
                config,
                (
                    F32Ptr(a as *const f32),
                    F32Ptr(b as *const f32),
                    N,
                    F32MutPtr(out as *mut f32),
                ),
            )
        }?;
    }

    // prefetch to CPU
    unsafe {
        cudarc::driver::sys::cuStreamAttachMemAsync(
            std::ptr::null_mut(),
            out,
            0,
            cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_HOST as u32,
        )
    };

    dev.synchronize()?;

    let view_out = unsafe { std::slice::from_raw_parts(out as *mut f32, N) };

    for f in view_out {
        assert!((f - 3.).abs() < MAX_ERR, "found valud of {}", f);
    }

    println!("sanity check:");
    for f in view_out.iter().take(10) {
        print!("{f} ");
    }
    println!("...");
    for f in view_out[N - 10..].iter() {
        print!("{f} ");
    }
    println!();

    Ok(())
}
