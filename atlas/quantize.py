import torch
from torch import nn
import constriction
import numpy as np

def grad_scale(x, scale):
    """LSQ gradient scaling: identity in the forward pass, multiplies the gradient by `scale`
    in the backward pass (stabilizes learning of the quantizer scale/offset params)."""
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001, use_grad_scale=False):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.learned = learned
        self.entropy_type = entropy_type
        self.use_grad_scale = use_grad_scale
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True) # gamma
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True) # beta

        self.weight = weight

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]  # per-channel min/max of the input
        scale = (t_max - t_min) / (self.qmax-self.qmin)
        # Guard against zero scale (e.g. when all values are identical)
        scale = torch.clamp(scale, min=1.0/self.qmax)
        self.beta.data = t_min.to(device)
        self.scale.data = scale.to(device)

    def forward(self, x):
        if self.learned:
            if self.use_grad_scale:
                # LSQ gradient scaling for the learned scale/offset. Off by default for
                # back-compat with the validated results — see docs/findings/atlas_code_audit.md.
                g = 1.0 / ((self.qmax * x.numel()) ** 0.5)
                s_scale, beta_scale = grad_scale(self.scale, g), grad_scale(self.beta, g)
            else:
                s_scale, beta_scale = self.scale, self.beta
            code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = (x * self.qmax).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant / self.qmax

        bits, entropy_loss = 0, 0
        if not self.training:
            bits = self.size(quant)
        return dequant, entropy_loss*self.weight, bits  # dequantized value, entropy loss, compressed size (bits)

    def size(self, quant):
        index_bits = 0
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant.int().flatten().tolist())
        index_bits += (get_np_size(compressed) * 8 + 64)
        index_bits += get_np_size(histogram_table) * 8
        index_bits += get_np_size(unique) * 8
        index_bits += self.scale.numel()*torch.finfo(self.scale.dtype).bits  # gamma (scale) bits
        index_bits += self.beta.numel()*torch.finfo(self.beta.dtype).bits  # beta bits
        return index_bits

    def compress(self, x):
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        return code.round(), code.round()* self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta


def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix)
    # unique values, their indices, data replaced with indices of unique values, counts of them
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    # find range and type (bits) of unique values and compress the values
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judge_type(min_value, max_value))
    # Handle single unique value: no entropy coding needed
    if len(unique) == 1:
        compressed = np.array([], dtype=np.uint32)
        return compressed, unique_counts, unique
    # compress data replaced with indices of unique values
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities,perfect=True)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    # Final compressed output
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique


def decompress_matrix_flatten_categorical(compressed, unique_counts, unique, num_symbols, shape):
    """Inverse of :func:`compress_matrix_flatten_categorical`.

    Decodes the ANS stream back to the original integer matrix. ``unique`` /
    ``unique_counts`` are the symbol table returned by the encoder, ``num_symbols`` is the
    flattened element count, and ``shape`` is the target output shape. Uses ``perfect=True``
    to match the encoder's model, and handles the single-unique-symbol stream (empty
    ``compressed``) that the encoder special-cases.
    """
    unique = np.asarray(unique)
    if len(unique) == 1:
        decoded = np.full(num_symbols, unique[0], dtype=unique.dtype)
        return decoded.reshape(shape)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities, perfect=True)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    indices = decoder.decode(entropy_model, num_symbols)
    return unique[indices].reshape(shape)


def judge_type(min, max):
    if min>=0:
        if max<=255:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32

def get_np_size(x):
    return x.size * x.itemsize
