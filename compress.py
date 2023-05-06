import pickle
import numpy as np
import bitshuffle
from dahuffman import HuffmanCodec
import time
import math

with open('frequency.pkl', 'rb') as f:
    frequency = pickle.load(f)
codec = HuffmanCodec.from_frequencies(frequency)

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


def compress(x, alg='lz4'):
    x = x.reshape(-1)
    x_q, x_max, x_min = linear_quantize(x)
    if alg=='lz4':
        x_c = bitshuffle.compress_lz4(x_q)
    elif alg=='huffman':
        x_c = codec.encode(x_q.astype('str').tolist())
    return x_c, x_max, x_min


def uncompress(x_c, x_max, x_min, x_shape, alg='lz4'):
    if alg=='lz4':
        x = bitshuffle.decompress_lz4(x_c, (math.prod(x_shape), ),
                                  np.dtype('uint8'))
    elif alg=='huffman':
        x = codec.decode_streaming(x_c)
        x = np.array(list(x)).astype(np.uint8)

    x = linear_dequantize(x, x_max, x_min)
    x = x.reshape(x_shape)
    return x

def benchmark():
    n = 16551
    ct = [[], [], []]
    ut = [[], [], []]
    ratio = [[], [], []]
    for i in range(3):
        for j in range(n):
            with open("inters/test/{}/{}.npy".format(i + 1, j), 'rb') as f:
                a = np.load(f)

            start = time.time()
            # a_c, a_max, a_min = compress(a)
            a_c, a_max, a_min = linear_quantize(a)
            end = time.time()
            ct[i].append(end - start)

            start = time.time()
            a_uncompressed = uncompress(a_c, a_max, a_min, a.shape)
            end = time.time()
            ut[i].append(end - start)

            # quantization psnr
            # print(np.allclose(a, a_uncompressed, atol=0.01))
            # print(psnr(a, a_dequantized))
            # print(a_quantized.nbytes / a.nbytes)
            # print(a_compressed.nbytes / a.nbytes)
            # print("compression ratio: {}".format(a.nbytes / a_c.nbytes))
            # print("compression ratio: {}".format(a.nbytes / len(a_c)))
            ratio[i].append(a.nbytes / len(a_c))
            break

    for i in range(3):
        print(sum(ct[i]) / len(ct[i]))
        print(sum(ut[i]) / len(ut[i]))
        print(sum(ratio[i]) / len(ratio[i]))
        print('\n')

if __name__ == "__main__":
    with open("inters/test/1/0.npy", 'rb') as f:
        a = np.load(f)
    
    a_c, a_max, a_min = compress(a, alg='huffman')
    a_uncompressed = uncompress(a_c, a_max, a_min, a.shape, alg='huffman')
    print(np.allclose(a, a_uncompressed, atol=0.01))
    # print("compression ratio: {}".format(a.nbytes / a_c.nbytes))
    print("compression ratio: {}".format(a.nbytes / len(a_c)))