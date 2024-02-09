use std::path::PathBuf;

const WIDTH: usize = 16;
const HEIGHT: usize = 16;

fn main() {
    let save_dir: PathBuf = PathBuf::from("./data");

    // directory to store sample images
    if let Err(e) = std::fs::create_dir(&save_dir) {
        match e.kind() {
            std::io::ErrorKind::PermissionDenied => {
                panic!("permission denied, can not create the directory")
            }
            std::io::ErrorKind::AlreadyExists => {
                println!("{} directory already exists.", save_dir.display())
            }
            e => panic!("{e}"),
        };
    }
    println!("saving data to: {}", save_dir.display());

    // test images to be:
    //
    // BLACK <--->  RED
    //   |           |
    //  BLUE <---> WHITE
    //

    // save an "RGRG" 8-bit test image
    let mut rgrg: Vec<u8> = Vec::with_capacity(HEIGHT * WIDTH);
    for row in (0..WIDTH).step_by(2) {
        // RGRG row
        for col in (0..HEIGHT).step_by(2) {
            let red: u8 = ((col * 255) as f64 / (WIDTH as f64)).floor() as u8;
            let green: u8 = (0.5 * ((col * 128) as f64 / (WIDTH as f64))
                + ((row * 128) as f64 / (HEIGHT as f64)))
                .floor() as u8;
            rgrg.push(red);
            rgrg.push(green);
        }

        // GBGB row
        for col in (0..HEIGHT).step_by(2) {
            let blue: u8 = ((row * 255) as f64 / (HEIGHT as f64)).floor() as u8;
            let green: u8 = (0.5 * ((col * 128) as f64 / (WIDTH as f64))
                + ((row * 128) as f64 / (HEIGHT as f64)))
                .floor() as u8;
            rgrg.push(green);
            rgrg.push(blue);
        }
    }
    println!("upper left:");
    println!("  {}, {}", rgrg[0], rgrg[1]);
    println!("  {}, {}", rgrg[WIDTH], rgrg[WIDTH + 1]);
    std::fs::write(save_dir.join("rgrg_8b.raw"), &rgrg).expect("failed to write file");
}
