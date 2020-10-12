import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import util

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq


def huffman_encode(arr, prefix, save_dir='./'):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)
    
    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32': float, 'int32': int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1
    
    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)
    
    # Merge nodes
    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')
    
    root = heappop(heap)
    generate_code(root, '')
    # print(root)
    # Path to save location
    directory = Path(save_dir)
    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    
    datasize = dump(data_encoding, directory/f'{prefix}.bin')
    
    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin')
    
    return treesize, datasize


def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)
    
    # Read the codebook
    codebook_encoding = load(directory/f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)
    
    # Read the data
    data_encoding = load(directory/f'{prefix}.bin')
    
    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None: # Leaf node
            data.append(ptr.value)
            ptr = root
    
    return np.array(data, dtype=dtype)


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr}
    code_list = []
    
    def encode_node(node):
        if node.value is not None: # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
    encode_node(root)

    return ''.join(code_list)


def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32':bitstr2float, 'int32':bitstr2int}
    idx = 0
    
    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = converter[dtype](code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()


# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
      
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    # print(filename)
    with open(filename, 'wb') as f:
        f.write(byte_arr)
    return len(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    # print(filename)
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's


def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]


def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's


def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]
  


def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])


def reconstruct_indptr_conv(data_size, length):
    x = [data_size for _ in range(data_size)]
    return np.concatenate([[0], np.cumsum(x)])

# Encode / Decode models
def huffman_encode_model(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    compressed_total = 0

    log_text = f"{'Layer':<15} | {'original bytes':>20} {'compressed bytes':>20} {'improvement':>11} {'percent':>7}"
    util.log(log_text)
    print(log_text)

    log_text = '-'*70
    util.log(log_text)
    print(log_text)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            if 'conv' in name:

                #################################
                # TODO:
                #   You can refer to the code of "fc" section below, but note that "csr_matrix" can only be used on
                #   2-dimensional data
                #   --------------------------------------------------------
                #   HINT:
                #   Suppose the shape of the weights of a certain convolution layer is (Kn, Ch, W, H)
                #   ---
                #   1. Call function "csr_matrix" for all (Kn * Ch) two-dimensional matrices (W, H), and get "data",
                #   "length of data", "indices", and "indptr" of all (Kn * Ch) csr_matrix.
                #   2. Concatenate these 4 parts of all (Kn * Ch) csr_matrices individually into 4 one-dimensional
                #   lists, so there will be 4 lists.
                #   3. Do huffman coding on these 4 lists individually.
                #################################

                # Note that we do not huffman encode "conv" yet. The following four lines of code need to be modified

                
                Kn, Ch = shape[0], shape[1]
                form = 'csr'

                data = []
                length_of_data = []
                indices = []
                indptr = [] 
        
                for i in range(Kn):
                    for j in range(Ch):
                        
                        mat = csr_matrix(weight[i][j])
                        
                        data.append(mat.data)
                        indices.append(mat.indices)
                        indptr.append(calc_index_diff(mat.indptr))
                        length_of_data.append(mat.data.size)
                
                all_data = np.concatenate(data)
                all_indices = np.concatenate(indices)
                all_indptr = np.concatenate(indptr).reshape(-1)
                len_array = np.array(length_of_data).reshape(-1)
                
                all_indptr = np.append(all_indptr, np.array(0).reshape(-1), axis=0)
                len_array = np.append(len_array, np.array(0).reshape(-1), axis=0)
                

                t0, d0 = huffman_encode(all_data, name+f'_{form}_data', directory)
                t1, d1 = huffman_encode(all_indices, name+f'_{form}_indices', directory)
                t2, d2 = huffman_encode(all_indptr, name+f'_{form}_indptr', directory)
                t3, d3 = huffman_encode(len_array, name + '_length', directory)
                
                

                original = param.data.cpu().numpy().nbytes
                compressed = t0 + t1 + t2  + d0 + d1 + d2 
                
                log_text = f"{name:<15} | {original:20} {compressed:20} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}% (NEED TO BE IMPLEMENTED)"
                util.log(log_text)
                print(log_text)

                # # Note that we do not huffman encode "conv" yet. The following four lines of code need to be modified
                # conv = param.data.cpu().numpy()
                # conv.dump(f'{directory}/{name}')

                # # Print statistics
                # original = conv.nbytes
                # compressed = original
                # log_text = f"{name:<15} | {original:20} {compressed:20} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}% (NEED TO BE IMPLEMENTED)"
                # util.log(log_text)
                # print(log_text)
            elif 'fc' in name:
                
                form = 'csr' if shape[0] < shape[1] else 'csc'
                mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
                # print(weight.shape)
                # print(mat.indptr.shape)
                # Encode
                
                t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory)
                t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory)
                t2, d2 = huffman_encode(calc_index_diff(mat.indptr), f'{form}_indptr', directory)

                # print(shape)
                # print(mat.indptr)

                # Print statistics
                original = param.data.cpu().numpy().nbytes
                compressed = t0 + t1 + t2 + d0 + d1 + d2
                log_text = f"{name:<15} | {original:20} {compressed:20} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%"
                util.log(log_text)
                print(log_text)
        else:  # bias
            # Note that we do not huffman encode bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}')

            # Print statistics
            original = bias.nbytes
            compressed = original

            log_text = f"{name:<15} | {original:20} {compressed:20} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%"
            util.log(log_text)
            print(log_text)
        original_total += original
        compressed_total += compressed

    log_text = '-'*70
    util.log(log_text)
    print(log_text)

    log_text = f"{'total':15} | {original_total:>20} {compressed_total:>20} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%"
    util.log(log_text)
    print(log_text)


def huffman_decode_model(model, directory='encodings/'):
    for name, param in model.named_parameters():
        if 'weight' in name:
            dev = param.device
            weight = param.data.cpu().numpy()
            shape = weight.shape
            if 'conv' in name:

                #################################
                # TODO:
                #   Decode according to the code of "conv" section you write in the function "huffman encode model"
                #   above, and refer to encode and decode code of "fc"
                #################################

                # Note that we do not huffman decode "conv" yet. The following three lines of code need to be modified
                
                # print(name, shape)

                Kn, Ch = shape[0], shape[1]
                form = 'csr'            

                data = huffman_decode(directory, name+f'_{form}_data', dtype='float32')
                indices = huffman_decode(directory, name+f'_{form}_indices', dtype='int32')
                # indptr = reconstruct_indptr(huffman_decode(directory, name+f'_{form}_indptr', dtype='int32'))
                
                indptr = huffman_decode(directory, name+f'_{form}_indptr', dtype='int32')
                data_length = huffman_decode(directory, name + '_length', dtype='int32')


                indptr = indptr[:-1]
                data_length = data_length[:-1]
                
                # print(indices.size)
                # print(data.size)
                # if 'conv5' in name: 
                #     print( reconstruct_indptr(indptr[56816*3-1:56817*3-1]) )

                # if 'conv5' in name:
                #     for i, j in enumerate(indptr):
                #         if j != 3:
                #             print(i)
                # if 'conv5' in name:
                #     print(indptr[170447])

                # print(indptr[:shape[2]])
                # print(reconstruct_indptr( indptr[:shape[2]]))
                # after_reconstruct = reconstruct_indptr_conv(indptr[0], indptr[-1])
                
                # print(data.size)
                # print(shape[2])
                start = 0 
                indptr_id = 0
                indptr_start = 0
                L = 0

                origin_array = []

                # if 'conv5' in name:
                #     print('**************')
                #     print(data_length[56815])
                #     print('**************')

                index = 0
                for i in range(Kn):
                    for j in range(Ch):
                        data_size = data_length[index]
                        index += 1
                        # arr = csr_matrix((data[start:start+data_size], indices[start:start+data_size], after_reconstruct), (shape[2], shape[3])).toarray()
                        # if 'conv5' in name:
                        #     print(data[start:start+data_size].size, reconstruct_indptr(indptr[indptr_id:indptr_id+shape[2]])[-1])
                        # if 'conv5' in name:
                        #     print('i+j', i+j)
                        #     if indptr[indptr_id:indptr_id+shape[2]][0] == 2:
                        #         print('correct!')
                        # if i+j == 510 and 'conv5' in name:
                        #     print(i, j) 
                        #     # i = 255 , j = 255 
                        #     print(data_size, shape[2])
                        #     print(reconstruct_indptr(indptr[indptr_id:indptr_id+shape[2]]))
                        #     print(len(indices[start:start+data_size]))
                        #     print('for testing')
                        #     print('indptr_id = ', indptr_id+shape[2]-1)
                        #     print('data = ', start+data_size-1)
                        #     print(indptr[indptr_id+shape[2]-1])
                        # if i+j == 510 and 'conv5' in name:
                        #     print(indptr[indptr_id+shape[2]])
                        arr = csr_matrix((data[start:start+data_size], indices[start:start+data_size], reconstruct_indptr(indptr[indptr_id:indptr_id+shape[2]])), (shape[2], shape[3])).toarray()
                        
                        # if reconstruct_indptr(indptr[indptr_id:indptr_id+shape[2]])[-1] != len(indices[start:start+data_size]):
                        #     print('find you!!')
                        #     print(i, j)
                        #     print(reconstruct_indptr(indptr[indptr_id:indptr_id+shape[2]]))
                        #     print(len(indices[start:start+data_size]))

                        start += data_size
                        indptr_id += shape[2]
                        origin_array.append(arr)

                origin_weight = np.concatenate(origin_array).reshape(shape[0],shape[1], shape[2], shape[3])

                # print(origin_weight.shape)

                param.data = torch.from_numpy(origin_weight).to(dev)

                # dev = param.device
                # conv = np.load(directory + '/' + name, allow_pickle=True)
                # param.data = torch.from_numpy(conv).to(dev)
            elif 'fc' in name:
                form = 'csr' if shape[0] < shape[1] else 'csc'
                matrix = csr_matrix if shape[0] < shape[1] else csc_matrix
                
                # Decode data
                data = huffman_decode(directory, name+f'_{form}_data', dtype='float32')
                indices = huffman_decode(directory, name+f'_{form}_indices', dtype='int32')
                indptr = reconstruct_indptr(huffman_decode(directory, name+f'_{form}_indptr', dtype='int32'))
                
                # Construct matrix
                mat = matrix((data, indices, indptr), shape)

                # Insert to model
                param.data = torch.from_numpy(mat.toarray()).to(dev)
        else:
            dev = param.device
            bias = np.load(directory+'/'+name, allow_pickle=True)
            param.data = torch.from_numpy(bias).to(dev)
