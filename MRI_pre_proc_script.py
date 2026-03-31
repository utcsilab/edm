# import sigpy as sp
import os
# import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
from tqdm import tqdm
from utils import ifftc, fftc
import json


# def resize(input, H,W):
#     center_H = input.shape[-2]//2
#     center_W = input.shape[-1]//2
#     H_h = H//2
#     W_h = W//2
#     print(H_h)
#     print(W_h)
#     img_out = input[..., center_H-H_h:center_H+H_h, center_W-W_h:center_W+W_h]
#     return img_out

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.

    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.

    Returns:
        array: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = np.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)


BRAIN_Min_slice = 0 #number of noisy slices to remove from the end of each volume
KNEE_Min_slice = 10 #number of noisy slices to remove from the end of each volume

BRAIN_Max_slice = -2 #number of noisy slices to remove from the end of each volume
KNEE_Max_slice = -4 #number of noisy slices to remove from the end of each volume

LENGTH = 320

brain_file_list = glob.glob("/data/fastmri/brain_coilcombined/train/*.h5")
knee_file_list = glob.glob('/data/fastmri/knee_coilcombined/train/*.h5')
file_list =  brain_file_list + knee_file_list
print(len(file_list))
# file_list.remove(file_list[105])
# file_list.remove(file_list[345])
# file_list.remove(file_list[368])
file_list.remove('/data/fastmri/brain_coilcombined/train/file_brain_AXT1_201_6002824.h5')

class_dict = {} #keys are relative filepaths to slices and vals are class number

save_root = "/data/edm_training_data/fastmri_all_preprocessed/"
count = 0
for fname in tqdm(file_list):
    with h5py.File(fname, 'r') as data:
        file_name = os.path.splitext(os.path.basename(fname))[0]
        #(1) Make the MVUE two-channel image and remove the noise slices
        try:
            mvue_vol = np.asarray(data['mvue_vol'])[:,0,...]
        except:
            print('Missing: ',fname)
            break

        b = mvue_vol.shape[0]
        H = mvue_vol.shape[-2]
        W = mvue_vol.shape[-1]
        mvue_vol = resize(mvue_vol, oshape=(b,W,W))
        mvue_ksp = fftc(mvue_vol, axes=(-2,-1))
        mvue_ksp = resize(mvue_ksp, oshape=(b,LENGTH,LENGTH))
        mvue_vol = ifftc(mvue_ksp, axes=(-2,-1))
    
        aqc = data.attrs['acquisition']
        # print(aqc)
        if 'T2' in aqc:
            MIN = BRAIN_Min_slice
            MAX = BRAIN_Max_slice
            cur_class = 0
            # print(aqc)
        if 'T1' in aqc:
            MIN = BRAIN_Min_slice
            MAX = BRAIN_Max_slice
            cur_class = 1
            # print(aqc)
        if 'FLAIR' in aqc:
            MIN = BRAIN_Min_slice
            MAX = BRAIN_Max_slice
            cur_class = 2
            # print(aqc)
        if 'CORPD_FBK' in aqc:
            MIN = KNEE_Min_slice
            MAX = KNEE_Max_slice
            cur_class = 3
            # print(aqc)
        if 'CORPDFS_FBK' in aqc:
            MIN = KNEE_Min_slice
            MAX = KNEE_Max_slice
            cur_class = 4
            # print(aqc)
        
        # reshaping stuff
        try:
            mvue_vol = mvue_vol[MIN:MAX]
        except:
            print('insufficient slices: ', fname)
            continue
        
        
        # go to two channels 
        two_channel_img = np.stack((mvue_vol.real, mvue_vol.imag), axis=1)
        # print(two_channel_img.shape)
        # break
        abs_imgs = abs(mvue_vol)
        norm_maxes = np.max(abs(abs_imgs), axis=(-2,-1), keepdims=True)
        try:
            normalised_slices = (two_channel_img) / (norm_maxes[...,None])
            # print(norm_maxes.shape)
            # print(normalised_slices.shape)
            # print(two_channel_img.shape)
            # if count ==105:
            #     print(fname)
        except:
            print('bad divide:', fname)
            continue

        if np.sum(norm_maxes)==0:
            print('re calc: ', fname)

        cur_path = save_root

        
        
        
        #(c) add patient ID to path
        cur_path = os.path.join(cur_path, file_name)

        # #NOTE debugging
        # print("Save Path: ", cur_path)
        # print("Class: ", cur_class)
        
        #(d) make the path and save the slices and metadata
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        
        for i in range(normalised_slices.shape[0]):
            slice_path = os.path.join(cur_path, str(i) + ".npy")
            
            np.save(slice_path, normalised_slices[i])
            
            relative_path = slice_path.split(save_root)[-1][0:]
            class_dict[relative_path] = cur_class
        
    


    count = count+1

json_output = {"labels": [[k, v] for k, v in class_dict.items()]}

j = json.dumps(json_output, indent=4)
with open(os.path.join(save_root, "dataset.json"), "w") as f:
    print(j, file=f) 