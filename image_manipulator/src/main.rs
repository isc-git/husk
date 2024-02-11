use std::{mem::size_of, path::PathBuf};

use cudarc::driver::{
    sys::{CUdeviceptr, CUmemAttach_flags},
    DeviceRepr, LaunchAsync,
};
use image::codecs::tiff::TiffEncoder;

const WIDTH: usize = 3840;
const HEIGHT: usize = 2160;
const DATA_DIR: &str = "./data";
const KERNEL_DIR: &str = "../kernels";

struct U8Ptr(*const u8);
struct U8MutPtr(*mut u8);

unsafe impl DeviceRepr for U8Ptr {}
unsafe impl DeviceRepr for U8MutPtr {}

fn main() -> anyhow::Result<()> {
    // configure CUDA device
    let gpu = cudarc::driver::CudaDevice::new(0)?;

    // compile kernels
    let kernel_dir = PathBuf::from(KERNEL_DIR);
    let debayer_ptx =
        cudarc::nvrtc::compile_ptx(std::fs::read_to_string(kernel_dir.join("debayer.cu"))?)?;

    // load ptx onto device
    gpu.load_ptx(debayer_ptx, "debayer", &["naive_debayer_kernel"])?;

    // configure directory to save data
    let save_dir: PathBuf = PathBuf::from(DATA_DIR);
    if let Err(e) = std::fs::create_dir(&save_dir) {
        match e.kind() {
            std::io::ErrorKind::PermissionDenied => {
                panic!("permission denied, can not create the directory")
            }
            std::io::ErrorKind::AlreadyExists => {
                println!("{} directory already existed.", save_dir.display())
            }
            e => panic!("{e}"),
        };
    }
    println!("saving data to: {}", save_dir.display());

    //  BLACK --> RED
    //  BLUE  --> WHITE
    let mut rgrg: Vec<u8> = Vec::with_capacity(HEIGHT * WIDTH);
    for row in (0..HEIGHT).step_by(2) {
        // RGRG row
        for col in (0..WIDTH).step_by(2) {
            let red: u8 = (((col as f64) / (WIDTH as f64)) * 255.).floor() as u8;
            let green: u8 =
                (((row as f64) / (HEIGHT as f64) + ((col as f64) / (WIDTH as f64))) * 0.5 * 255.)
                    .floor() as u8;
            rgrg.push(red);
            rgrg.push(green);
        }

        // GBGB row
        for col in (0..WIDTH).step_by(2) {
            let blue: u8 = (((row as f64) / (HEIGHT as f64)) * 255.).floor() as u8;
            let green: u8 =
                (((row as f64) / (HEIGHT as f64) + ((col as f64) / (WIDTH as f64))) * 0.5 * 255.)
                    .floor() as u8;
            rgrg.push(green);
            rgrg.push(blue);
        }
    }

    let raw_tiff = std::fs::File::create(save_dir.join("raw_rgrg.tiff"))?;
    let raw_tiff_w = std::io::BufWriter::new(raw_tiff);
    let encoder = TiffEncoder::new(raw_tiff_w);
    encoder.encode(&rgrg, WIDTH as u32, HEIGHT as u32, image::ColorType::L8)?;

    /*
     * Basic debayering
     */
    let rgb_height = HEIGHT / 2;
    let rgb_width = WIDTH / 2;
    let rgb_bpp = 4; // RGBA
    let rgb_size = rgb_width * rgb_height * rgb_bpp * size_of::<u8>();

    // using managed memory for iGPU
    let mut rgrg_managed: CUdeviceptr = 0;
    unsafe {
        cudarc::driver::sys::cuMemAllocManaged(
            &mut rgrg_managed as *mut CUdeviceptr,
            size_of::<u8>() * rgrg.len(),
            CUmemAttach_flags::CU_MEM_ATTACH_HOST as u32,
        );
    }
    let mut rgb: CUdeviceptr = 0;
    unsafe {
        cudarc::driver::sys::cuMemAllocManaged(
            &mut rgb as *mut CUdeviceptr,
            rgb_size,
            CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        );
    }

    unsafe {
        let view_rgrg_managed = std::slice::from_raw_parts_mut(rgrg_managed as *mut u8, rgb_size);
        view_rgrg_managed[..rgb_size].clone_from_slice(&rgrg);
    }

    let block_size: u32 = 256;
    let num_blocks: u32 = ((size_of::<u8>() * rgrg.len()) as u32 + block_size - 1) / block_size;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // prefetching hints for performance improvements
    unsafe {
        cudarc::driver::sys::cuStreamAttachMemAsync(
            std::ptr::null_mut(),
            rgrg_managed,
            0,
            CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        )
    };

    let debayer = gpu
        .get_func("debayer", "naive_debayer_kernel")
        .expect("could not find debayer_kernel");
    unsafe {
        debayer.launch(
            config,
            (
                U8Ptr(rgrg_managed as *const u8),
                WIDTH,
                HEIGHT,
                U8MutPtr(rgb as *mut u8),
            ),
        )
    }?;

    // prefetch to CPU
    unsafe {
        cudarc::driver::sys::cuStreamAttachMemAsync(
            std::ptr::null_mut(),
            rgb,
            0,
            CUmemAttach_flags::CU_MEM_ATTACH_HOST as u32,
        )
    };

    gpu.synchronize()?;

    let view_out = unsafe { std::slice::from_raw_parts(rgb as *mut u8, rgb_size) };

    let basic_tiff = std::fs::File::create(save_dir.join("basic_rgb.tiff"))?;
    let basic_tiff_w = std::io::BufWriter::new(basic_tiff);
    let encoder = TiffEncoder::new(basic_tiff_w);
    encoder.encode(
        view_out,
        rgb_width as u32,
        rgb_height as u32,
        image::ColorType::Rgba8,
    )?;

    Ok(())
}
