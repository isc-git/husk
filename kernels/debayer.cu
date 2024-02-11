static __device__ uchar4 
naive_bayer2rgba(unsigned char quad[4])
{
    uchar4 rgba;
    rgba.x = quad[0];
    rgba.y = (quad[1] / 2 + quad[2] / 2);
    rgba.z = quad[3];
    rgba.w = 255;

    return rgba;
}

// Demosaics a Bayer buffer into an RGBA output.
extern "C" __global__ void
naive_debayer_kernel(
    unsigned char* bayer,
    int bayer_width,
    int bayer_height,
    uchar4* rgba
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int step_x = blockDim.x * gridDim.x;
    int step_y = blockDim.y * gridDim.y;

    int rgba_width = bayer_width / 2;
    int rgba_height = bayer_height / 2;

    for (int col = x; col < rgba_width; col += step_x)
    {
        for (int row = y; row < rgba_height; row += step_y)
        {
            unsigned char* offset = bayer + (col * 2) + (row * bayer_width * 2);
            unsigned char quad[4];
            quad[0] = *(offset);
            quad[1] = *(offset + 1);
            quad[2] = *(offset + bayer_width);
            quad[3] = *(offset + bayer_width + 1);

            uchar4 rgba_quad = naive_bayer2rgba(quad);

            rgba[rgba_width * row + col] = rgba_quad;
        }
    }
}

//// Demosaic with basic linear interpolation
//extern "C" __global__ void
//bilinear_debayer_kernel(
//    unsigned char *bayer,
//    int width,
//    int height,
//    uchar* rgba,
//) {
//
//}
