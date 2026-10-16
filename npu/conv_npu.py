import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


PARTITION_DIM = nl.tile_size.pmax
STATIONARY_FREE = nl.tile_size.gemm_stationary_fmax
MOVING_FREE_MAX = nl.tile_size.gemm_moving_fmax


def _preload_weights(W, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width):
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
            # Contiguous load: partition = c_out, free = (c_in, kh, kw)
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
                    # Strided read is fine in SBUF: (c_out, c_in)
                    w_tap = nl.copy(w_raw[:, :, kh, kw])
                    # Transpose to (c_in, c_out) so c_in is the contraction/partition dim
                    w_t = nisa.nc_transpose(w_tap)
                    w_sbuf[kh, kw, oc, ic] = nl.copy(w_t, dtype=W.dtype)

    return w_sbuf


def _add_bias(row, bias_2d, oc):
    """Broadcast one output-channel bias tile across the row width.
    """
    bias_col = nl.load(
        bias_2d[oc * PARTITION_DIM:(oc + 1) * PARTITION_DIM, :]
    )
    return nl.add(row, bias_col)


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
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % PARTITION_DIM == 0
    assert out_channels % PARTITION_DIM == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert MOVING_FREE_MAX >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = PARTITION_DIM
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // PARTITION_DIM
    w_sbuf = _preload_weights(
        W, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width
    )
    bias_2d = bias.reshape((out_channels, 1))

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for oc in nl.affine_range(n_tiles_c_out):
            for oh in nl.affine_range(out_height):
                # PSUM accumulates in float32. Copy casts back to X.dtype.
                res_psum = nl.zeros(
                    (PARTITION_DIM, out_width), nl.float32, buffer=nl.psum
                )

                for kh in nl.affine_range(filter_height):
                    for kw in nl.affine_range(filter_width):
                        for ic in nl.affine_range(n_tiles_c_in):
                            # Shifted input row: pixel (oh, ow) reads X[:, oh + kh, ow + kw].
                            x_tile = nl.ndarray(
                                shape=(nl.par_dim(c_in_pmax), out_width),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            x_tile[...] = nl.load(
                                X[
                                    b,
                                    ic * c_in_pmax:(ic + 1) * c_in_pmax,
                                    oh + kh,
                                    kw:kw + out_width,
                                ]
                            )
                            res_psum += nl.matmul(
                                w_sbuf[kh, kw, oc, ic],
                                x_tile,
                                transpose_x=True,
                            )

                row = nl.copy(res_psum, dtype=X.dtype)
                row = _add_bias(row, bias_2d, oc)
                nl.store(
                    X_out[
                        b,
                        oc * PARTITION_DIM:(oc + 1) * PARTITION_DIM,
                        oh,
                        0:out_width,
                    ],
                    value=row,
                )

    return X_out
