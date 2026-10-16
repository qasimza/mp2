import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


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

@nki.jit
def conv2d(X, W, bias):

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
    assert in_channels % 128 == 0
    assert out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_in_pmax
    output_rows_per_matmul = (nl.tile_size.gemm_moving_fmax + filter_width - 1) // input_width
    output_rows_per_strip = 32 # Random selection
    X_flat = X.reshape((batch_size, in_channels, input_height * input_width))

    # Reserve sapce in SBUF for bias
    bias_sbuf = nl.ndarray(
        shape=(
            n_tiles_c_out, 
            nl.par_dim(c_in_pmax), 
            1
            ), 
        dtype=bias.dtype, 
        buffer=nl.sbuf
        )

    # Reserve sapce in SBUF for every weight tile
    w_sbuf = nl.ndarray(
        shape=(
            filter_height, 
            filter_width, 
            n_tiles_c_out, 
            n_tiles_c_in, 
            nl.par_dim(c_in_pmax), 
            c_in_pmax
        ), 
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    # Loading weight tiles and bias
    for oc_tile_idx in nl.affine_range(n_tiles_c_out):

        # Bias into sbuf
        bias_sbuf[oc_tile_idx] = nl.load(bias[oc_tile_idx * c_in_pmax:(oc_tile_idx + 1) * c_in_pmax])

        # Weight tiles into sbuf
        for ic_tile_idx in nl.affine_range(n_tiles_c_in):

            w_raw = nl.ndarray(
                shape=(
                    nl.par_dim(c_in_pmax), 
                    c_in_pmax, 
                    filter_height, 
                    filter_width
                    ), 
                dtype=W.dtype, 
                buffer=nl.sbuf)
            w_raw[...] = nl.load(W[oc_tile_idx * c_in_pmax:(oc_tile_idx + 1) * c_in_pmax, ic_tile_idx * c_in_pmax:(ic_tile_idx + 1) * c_in_pmax, :, :])

            for filter_height_idx in nl.affine_range(filter_height):
                for filter_width_idx in nl.affine_range(filter_width):
                    w_slice = nl.copy(w_raw[:, :, filter_height_idx, filter_width_idx])
                    w_slice_transposed = nisa.nc_transpose(w_slice)
                    w_sbuf[filter_height_idx, filter_width_idx, oc_tile_idx, ic_tile_idx] = nl.copy(w_slice_transposed, dtype=W.dtype)

    # Process the images in batches 
    for b in nl.affine_range(batch_size):
        
        for first_row in range(0, out_height, output_rows_per_strip): 
            
            num_rows = min(output_rows_per_strip, out_height - first_row)
            num_input_rows = num_rows + filter_height - 1
            
            # Reserve space for this strip's input rows
            x_strip_sbuf = nl.ndarray(
                shape=(
                    n_tiles_c_in, 
                    nl.par_dim(c_in_pmax), 
                    num_input_rows * input_width
                    ), 
                dtype=X.dtype, 
                buffer=nl.sbuf
            )

            # Load input rows into SBUF
            for ic_tile_idx in nl.affine_range(n_tiles_c_in):
                x_strip_sbuf[ic_tile_idx] = nl.load(X_flat[b, ic_tile_idx * c_in_pmax:(ic_tile_idx + 1) * c_in_pmax, first_row * input_width:(first_row + num_input_rows) * input_width])


            # For each output block compute rows in SBUF
            for oc_tile_idx in nl.affine_range(n_tiles_c_out):
                o_strip_sbuf = nl.ndarray(
                    shape=(
                        nl.par_dim(c_in_pmax),
                        num_rows, 
                        out_width
                    ), 
                    dtype=X.dtype, 
                    buffer=nl.sbuf
                )

                for group_first_row in range(0, num_rows, output_rows_per_matmul):
                    rows_in_group = min(output_rows_per_matmul, num_rows - group_first_row)
                    num_columns = rows_in_group * input_width - (filter_width - 1)

                    # Running total for this group, in PSUM
                    res_psum = nl.zeros((c_in_pmax, num_columns), nl.float32, buffer=nl.psum)

                    for filter_height_idx in nl.affine_range(filter_height):
                        for filter_width_idx in nl.affine_range(filter_width):
                            for ic_tile_idx in nl.affine_range(n_tiles_c_in):
                                start = (group_first_row + filter_height_idx) * input_width + filter_width_idx
                                res_psum += nl.matmul(
                                    w_sbuf[filter_height_idx, filter_width_idx, oc_tile_idx, ic_tile_idx],
                                    x_strip_sbuf[ic_tile_idx][:, start:start + num_columns],
                                    transpose_x=True,
                                )

                    # Move from PSUM to SBUF after adding bias
                    for row_in_group in range(rows_in_group):
                        o_strip_sbuf[:, group_first_row + row_in_group, :] = nl.add(
                            res_psum[:, row_in_group * input_width:row_in_group * input_width + out_width],
                            bias_sbuf[oc_tile_idx],
                            dtype=X.dtype,
                        )

                # One store per (strip, output-channel tile)
                nl.store(
                    X_out[b, oc_tile_idx * c_in_pmax:(oc_tile_idx + 1) * c_in_pmax,
                          first_row:first_row + num_rows, :],
                    value=o_strip_sbuf,
                )

    return X_out

