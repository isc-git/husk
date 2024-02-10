use cudarc::driver::LaunchAsync;

const N: usize = 10_000_000;
const BENCH: usize = 1000;
const MAX_ERR: f32 = 1e-9;

fn main() -> anyhow::Result<()> {
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let ptx = cudarc::nvrtc::compile_ptx(include_str!("../vector_add.cu"))?;
    println!("compiled ptx");

    // load ptx onto device
    dev.load_ptx(ptx, "adder", &["vector_add"])?;
    println!("loaded ptx onto device");

    // allocate host vectors
    let a = vec![1.0f32; N];
    let b = vec![2.0f32; N];
    let mut out = vec![0.0f32; N];

    // allocating device output
    let mut d_out = unsafe { dev.alloc::<f32>(N) }?;
    let mut d_a = unsafe { dev.alloc::<f32>(N) }?;
    let mut d_b = unsafe { dev.alloc::<f32>(N) }?;

    let block_size: u32 = 256;
    let num_blocks: u32 = (N as u32 + block_size - 1) / block_size;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..BENCH {
        // copy vectors to device
        dev.htod_sync_copy_into(&a, &mut d_a)?;
        dev.htod_sync_copy_into(&b, &mut d_b)?;

        let vector_add = dev.get_func("adder", "vector_add").unwrap();
        unsafe { vector_add.launch(config, (&d_a, &d_b, N, &mut d_out)) }?;

        dev.dtoh_sync_copy_into(&d_out, &mut out)?;
    }

    out.iter()
        .for_each(|f| assert!((f - 3.).abs() < MAX_ERR, "found value of {f}"));

    println!("sanity check:");
    for f in out.iter().take(10) {
        print!("{f} ");
    }
    println!("...");
    for f in out[N - 10..].iter() {
        print!("{f} ");
    }
    println!();

    println!("Hello, world!");

    Ok(())
}
