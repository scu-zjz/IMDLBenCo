import numpy as np
from IMDLBenCo.datasets.utils import read_jpeg_from_memory
from IMDLBenCo.registry import POSTFUNCS

@POSTFUNCS.register_module()
def cat_net_post_func(data_dict):
    tp_img = data_dict['image']
    DCT_coef, qtables = __get_jpeg_info(tp_img)
    data_dict['DCT_coef'] = DCT_coef[0]
    data_dict['qtables'] = qtables[0]
    
def __get_jpeg_info(image_tensor):
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
    """
    num_channels = 1
    jpeg = read_jpeg_from_memory(image_tensor)

    # determine which axes to up-sample
    ci = jpeg.comp_info
    need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    # up-sample DCT coefficients to match image size
    DCT_coef = []
    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
        # case 1: row scale (O) and col scale (O)
        if need_scale[i][0]==1 and need_scale[i][1]==1:
            out_arr = np.zeros((r * 2, c * 2))
            out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

        # case 2: row scale (O) and col scale (X)
        elif need_scale[i][0]==1 and need_scale[i][1]==2:
            out_arr = np.zeros((r * 2, c))
            DCT_coef.append(out_arr)
            out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view[:, :, :, :]
            out_view[1::2, :, :, :] = coef_view[:, :, :, :]

        # case 3: row scale (X) and col scale (O)
        elif need_scale[i][0]==2 and need_scale[i][1]==1:
            out_arr = np.zeros((r, c * 2))
            out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view[:, :, :, :]
            out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

        # case 4: row scale (X) and col scale (X)
        elif need_scale[i][0]==2 and need_scale[i][1]==2:
            out_arr = np.zeros((r, c))
            out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view[:, :, :, :]

        else:
            raise KeyError("Something wrong here.")

        DCT_coef.append(out_arr)

    # quantization tables
    qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float64) for i in range(num_channels)]

    return DCT_coef, qtables
