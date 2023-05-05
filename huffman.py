import torch
import struct
import numpy as np
import dahuffman

def int8_to_bytes(x): # x is a integer and should in [0, 256)
    return struct.pack('>B', x)

def float_to_bytes(x): # x is a float
    return struct.pack('>f', x)

def int_to_bytes(x, bits): # large int to byte array
    padding = 0 if bits % 8 == 0 else 8 - bits % 8
    bits += padding
    x <<= padding
    mask = (1 << 8) - 1
    res = bytes()
    for i in range(8, bits+1, 8):
        res += int8_to_bytes(x >> (bits - i) & mask)
#         print(bytes_to_bin(int8_to_bytes(x >> (bits - i) & mask)))
    return res

def bytes_to_int8(x): 
    return struct.unpack('>B', x)[0]

def bytes_to_float(x):
    return struct.unpack('>f', x)[0]
 
def bytes_to_bin(x):
    return list(bin(byte)[2:].rjust(8, '0') for byte in x)

def get_comp_code(table, x):
    res = bytes()
    total_bits = 0
    buffer = 0
    for e in x:
        b, v = table.get(e, (0, 0))
        total_bits += b
        buffer = (buffer << b) + v
        while total_bits >= 8:
            byte = buffer >> (total_bits - 8)
            res += int8_to_bytes(byte)
            buffer = buffer - (byte << (total_bits - 8))
            total_bits -= 8
    if total_bits > 0:
        padding = 8 - total_bits
        buffer <<= padding
        res += int8_to_bytes(buffer)
    return res

def huffman_encode(x, bits):
    # bit...,  val..., compressed

    codec = dahuffman.HuffmanCodec.from_data(x)               
    table = codec.get_code_table()
#     codec.print_code_table()
    res = bytes()
    for i in range(2 ** bits):
        b, v = table.get(i, (0, -1))
        res += int8_to_bytes(b)
    res += get_comp_code(table, range(2 ** bits))
    res += get_comp_code(table, x)
    return res

def huffman_decode(x, bits, total):
    
    b_list = [bytes_to_int8(x[i:i+1]) for i in range(2 ** bits)]
    Cur = 2**bits
   
    v_list = []
    buffer = 0
    size = 0
    for b in b_list:
        while size < b:
            buffer = (buffer << 8) + bytes_to_int8(x[Cur:Cur+1])
            Cur += 1
            size += 8
        v = buffer >> (size - b)
        buffer -= v << (size - b)
        size -= b
        v_list.append(v)
   
    lookup = {}
    for i in range(2 ** bits):
        lookup[(v_list[i], b_list[i])] = i
    
    x_list = []
    buffer = 0
    size = 0
    for byte in x[Cur:]:
        for m in [128, 64, 32, 16, 8, 4, 2, 1]:
            buffer = (buffer << 1) + bool(byte & m)
            size += 1
            if (buffer, size) in lookup:
                x_list.append(lookup[(buffer, size)])
                buffer = 0
                size = 0
                if len(x_list) == total:
                    break
        if len(x_list) == total:
            break
    return np.array(x_list)
    

def encode(x, bits=8):  
    # encode numpy.ndarray(numpy.float32) to bytes
    # note that every dim of x should not beyond 255, otherwise you will need to modify the code accordingly.
    res = bytes()
    
    for e in x.shape:
        res += int8_to_bytes(e)
    
    x = x.ravel()
    min_ = x.min()
    max_ = x.max()
    res += float_to_bytes(min_)
    res += float_to_bytes(max_)

    x = ((x - min_) * (2 ** bits - 1) / (max_ - min_)).round().astype(np.int32)
    res += huffman_encode(x, bits)
    
    return res
    
def decode(x, bits=8):
    Cur = 0
    shape = tuple((bytes_to_int8(x[i:i+1]) for i in range(Cur, Cur+4)))
    Cur += 4
    total = 1
    for d in shape: total *= d
    min_ = bytes_to_float(x[Cur:Cur+4])
    Cur += 4
    max_ = bytes_to_float(x[Cur:Cur+4])
    Cur += 4
    x = huffman_decode(x[Cur:], bits, total)
    x = x.astype(np.float32)
    x = x * (max_ - min_) / (2 ** bits - 1) + min_
    x = x.reshape(shape)
    return x 

def test():
    c_bit = 8
    a = torch.randn(size=(20,20,20,20))
    x = a.cpu().numpy()
    y = encode(x, c_bit)
    print('original bytes:', a.nelement() * 4)
    print('compressed bytes:', len(y))
    print('compress ratio:', len(y) / (a.nelement() * 4))
    z = decode(y, c_bit)
    d = z - x
    print('==check the compress loss==')
    print('min_diff:',d.min(), 'max_diff:', d.max())
