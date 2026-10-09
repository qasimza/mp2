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
        Hint: Instead of torch tensors, 
        use of jnp arrays is required to leverage JIT compilation and 
        GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    patches = []
    for kh in range(KH):
        for kw in range(KW):
            patch = x_pad[:, :, kh : kh + out_h * S : S, kw : kw + out_w * S : S]
            patches.append(patch)

    stacked = jnp.stack(patches, axis=2)
    permuted = jnp.transpose(stacked, (0, 3, 4, 1, 2))
    cols = permuted.reshape(N, out_h * out_w, C * KH * KW)
    return cols

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, 
        use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, 
        so you cannot do indexing like out[i:j, :] = ... inside a JIT. 
        You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1
    
    # 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # 2) flatten self.weight into shape (C_out, C*KH*KW).
    flattened_weight = weight.reshape(C_out, C * KH * KW)

    # 3) perform tiled matmul after required reshaping is done.
    L = cols.shape[1]
    K = cols.shape[2]
    tile = 16
    weight_t = flattened_weight.T
    out = jnp.zeros((N, L, C_out), dtype=cols.dtype)
    for i in range(0, L, tile):
        i_end = min(i + tile, L)
        for j in range(0, C_out, tile):
            j_end = min(j + tile, C_out)
            acc = jnp.zeros((N, i_end - i, j_end - j), dtype=cols.dtype)
            for k in range(0, K, tile):
                k_end = min(k + tile, K)
                acc = acc + jnp.matmul(cols[:, i:i_end, k:k_end], weight_t[k:k_end, j:j_end])
            out = out.at[:, i:i_end, j:j_end].set(acc)

    # 4) Add bias.
    out = out + bias

    # 5) reshape output into shape (N, C_out, out_h, out_w).
    out = out.reshape(N, out_h, out_w, C_out)
    out = jnp.transpose(out, (0, 3, 1, 2))
    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 25, 25
    model = ConvModel(H, W, in_channels=3, out_channels=4, kernel_size=6, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(4, 3, H, W)

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

    jax.profiler.start_trace(logdir)
    with jax.profiler.TraceAnnotation("compile"):
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax).block_until_ready()
    with jax.profiler.TraceAnnotation("conv"):
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax).block_until_ready()
    jax.profiler.stop_trace()

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
