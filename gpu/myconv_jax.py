import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    patches = []
    for kh in range(KH):
        for kw in range(KW):
            patch = x_pad[:, :, kh:kh + S * out_h:S, kw:kw + S * out_w:S]
            patches.append(patch)
    
    patches = jnp.stack(patches, axis=2)  # shape: (N, C, KH*KW, out_h, out_w)
    patches = jnp.reshape(patches, (N, C, KH, KW, out_h, out_w))
    patches = jnp.transpose(patches, (0, 4, 5, 1, 2, 3))  # shape: (N, out_h, out_w, C, KH, KW)
    patches = jnp.reshape(patches, (N, out_h * out_w, C * KH * KW))
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1
    
    # 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # 2) flatten self.weight into shape (C_out, C*KH*KW).
    weights_flat = jnp.reshape(weight, (C_out, C * KH * KW))

    # 3) perform tiled matmul after required reshaping is done.
    weights_t = jnp.transpose(weights_flat)  # shape: (C*KH*KW, C_out)
    tile_size = 64
    output_tiles = []
    L = cols.shape[1]  # out_h * out_w
    for start in range(0, L, tile_size):
        end = min(start + tile_size, L)
        cols_tile = cols[:, start:end, :]  # shape: (N, tile_size, C*KH*KW)
        out_tile = jnp.matmul(cols_tile, weights_t)  # shape: (N, tile_size, C_out)
        output_tiles.append(out_tile)
    out = jnp.concatenate(output_tiles, axis=1)  # shape: (N, out_h*out_w, C_out)

    # 4) Add bias.
    out += jnp.reshape(bias, (1, 1, C_out))

    # 5) reshape output into shape (N, C_out, out_h, out_w).
    out = jnp.reshape(out, (N, out_h, out_w, C_out))
    out = jnp.transpose(out, (0, 3, 1, 2))  # shape: (N, C_out, out_h, out_w)

    #return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(out_jax, conv_ref, atol=1e-1))
