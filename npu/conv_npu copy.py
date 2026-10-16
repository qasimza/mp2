import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


PARTITION_DIM = nl.tile_size.pmax
STATIONARY_FREE = nl.tile_size.gemm_stationary_fmax
MOVING_FREE_MAX = nl.tile_size.gemm_moving_fmax


def load_weights(W, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width):
    """Load every filter tap into SBUF as a (K=c_in, M=c_out) tile."""
    w_sbuf = nl.ndarray(
        shape=(
            filter_height,
            filter_width,
            n_tiles_c_out,
            n_tiles_c_in,
            nl.par_dim(PARTITION_DIM),
            STATIONARY_FREE,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )

    for oc in nl.affine_range(n_tiles_c_out):
        for ic in nl.affine_range(n_tiles_c_in):
            
            w_raw = nl.ndarray(
                shape=(nl.par_dim(PARTITION_DIM), PARTITION_DIM,
                       filter_height, filter_width),
                dtype=W.dtype,
                buffer=nl.sbuf,
            )
            w_raw[...] = nl.load(
                W[
                    oc * PARTITION_DIM:(oc + 1) * PARTITION_DIM,
                    ic * PARTITION_DIM:(ic + 1) * PARTITION_DIM,
                    :,
                    :,
                ]
            )

            for kh in nl.affine_range(filter_height):
                for kw in nl.affine_range(filter_width):
                 
                    w_tap = nl.copy(w_raw[:, :, kh, kw])
                    
                    w_t = nisa.nc_transpose(w_tap)
                    w_sbuf[kh, kw, oc, ic] = nl.copy(w_t, dtype=W.dtype)

    return w_sbuf


@nki.jit
def conv2d(X, W, bias):
    """
    A convolution kernel that you need to implement.

    Parameters:
        X: the input tensor
        W: the weights of the convolution filters.
        bias: the biases of the convolution filters.

    expect: X.shape == [batch_size, in_channels, input_height, input_width]
    expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
    expect: bias.shape == [out_channels]
    expect: filter_height == filter_width
    expect: input_channels % 128 == 0
    expect: output_channels % 128 == 0

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width

    The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, _, filter_height, filter_width = W.shape
    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    assert in_channels % PARTITION_DIM == 0 and out_channels % PARTITION_DIM == 0
    n_ic = in_channels // PARTITION_DIM
    n_oc = out_channels // PARTITION_DIM

    R = (MOVING_FREE_MAX + filter_width - 1) // input_width
    assert R >= 1
    BAND = 32 

    X_out = nl.ndarray(
        (batch_size, out_channels, out_height, out_width),
        dtype=X.dtype, buffer=nl.hbm,
    )
    X_flat = X.reshape((batch_size, in_channels, input_height * input_width))

    w_sbuf = load_weights(W, n_oc, n_ic, filter_height, filter_width)

    bias_2d = bias.reshape((out_channels, 1))
    bias_sb = nl.ndarray((n_oc, nl.par_dim(PARTITION_DIM), 1), dtype=bias.dtype, buffer=nl.sbuf)
    for oc in nl.affine_range(n_oc):
        bias_sb[oc] = nl.load(bias_2d[oc * PARTITION_DIM:(oc + 1) * PARTITION_DIM, :])

    bands = [(r0, min(BAND, out_height - r0)) for r0 in range(0, out_height, BAND)]

    for b in range(batch_size):
        for (row0, nrows) in bands:
            in_rows = nrows + filter_height - 1
            groups = [(r0, min(R, nrows - r0)) for r0 in range(0, nrows, R)]

            x_band = nl.ndarray(
                (n_ic, nl.par_dim(PARTITION_DIM), in_rows * input_width),
                dtype=X.dtype, buffer=nl.sbuf,
            )
            for ic in nl.affine_range(n_ic):
                x_band[ic] = nl.load(
                    X_flat[b, ic * PARTITION_DIM:(ic + 1) * PARTITION_DIM,
                           row0 * input_width:(row0 + in_rows) * input_width]
                )

            for oc in nl.affine_range(n_oc):
                o_sb = nl.ndarray(
                    (nl.par_dim(PARTITION_DIM), nrows, out_width), dtype=X.dtype, buffer=nl.sbuf
                )
                for (r0, r) in groups:
                    N = r * input_width - (filter_width - 1)
                    res_psum = nl.zeros((PARTITION_DIM, N), nl.float32, buffer=nl.psum)

                    for kh in nl.affine_range(filter_height):
                        for kw in nl.affine_range(filter_width):
                            for ic in nl.affine_range(n_ic):
                                start = (r0 + kh) * input_width + kw
                                res_psum += nl.matmul(
                                    w_sbuf[kh, kw, oc, ic],
                                    x_band[ic][:, start:start + N],
                                    transpose_x=True,
                                )

                    for i in range(r):
                        o_sb[:, r0 + i, :] = nl.add(
                            res_psum[:, i * input_width:i * input_width + out_width],
                            bias_sb[oc],
                            dtype=X.dtype,
                        )

                nl.store(
                    X_out[b, oc * PARTITION_DIM:(oc + 1) * PARTITION_DIM, row0:row0 + nrows, :],
                    value=o_sb,
                )

    return X_out