import numpy as np
import bitshuffle
import time
import math


def linear_quantize(x):
    # Compute the scale factor and zero point
    x_min, x_max = np.min(x), np.max(x)
    scale = (x_max - x_min) / 255

    # Quantize the input data
    x_q = np.round(x / scale + x_min).astype(np.uint8)

    return x_q, x_max, x_min


def linear_dequantize(x_q, x_max, x_min):
    scale = (x_max - x_min) / 255
    x = x_min + scale * x_q
    return x


def psnr(a, b):
    """Compute PSNR between two 1-D numpy arrays"""
    mse = np.mean((a - b)**2)
    max_val = np.max(a)
    return 10 * np.log10((max_val**2) / mse)


def compress(x):
    x.reshape(-1)
    x_q, x_max, x_min = linear_quantize(x)
    x_c = bitshuffle.compress_lz4(x_q)
    return x_c, x_max, x_min


def uncompress(x_c, x_max, x_min, x_shape):
    x = bitshuffle.decompress_lz4(x_c, (math.prod(x_shape), ), np.dtype('uint8'))
    x = linear_dequantize(x, x_max, x_min)
    x = x.reshape(x_shape)
    return x


if __name__ == "__main__":
    with open('t.npy', 'rb') as f:
        a = np.load(f)

    print(a.nbytes)

    start = time.time()
    a_c, a_max, a_min = compress(a)
    end = time.time()
    print(end - start)

    # uncompress+dequantize
    start = time.time()
    a_uncompressed = uncompress(a_c, a_max, a_min, a.shape)
    end = time.time()
    print(end - start)

    # quantization psnr
    print(np.allclose(a, a_uncompressed, atol=0.01))
    # print(psnr(a, a_dequantized))
    # print(a_quantized.nbytes / a.nbytes)
    # print(a_compressed.nbytes / a.nbytes)
    print("compression ratio: {}".format(a.nbytes / a_c.nbytes))
