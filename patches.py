import numpy as np
import pdb
from helper import print_red
import h5py
import itertools
# from tqdm import tqdm
from tqdm.notebook import tqdm

def _patching_autofit(img_shape, patch_shape):
    '''
    Autofit patching strategy:
        Symmetrically cover the image with patches without beyond boundary parts as much as possible.
    img_shape: numpy.ndarray; shape = (3,)
    patch_shape: numpy.ndarray; shape = (3,)
    '''
    n_dim = len(img_shape)
    n_patches = np.ceil(img_shape / patch_shape)
    start = np.zeros(n_dim)
    step = np.zeros(n_dim)
    for dim in range(n_dim):
        if n_patches[dim] == 1:
            start[dim] = -(patch_shape[dim] - img_shape[dim])//2
            step[dim] = patch_shape[dim]
        else:
            overlap = np.ceil(n_patches[dim] * patch_shape[dim] - img_shape[dim])/(n_patches[dim] - 1)
            overflow = n_patches[dim] * patch_shape[dim] - (n_patches[dim] - 1) * overlap - img_shape[dim]
            start[dim] = - overflow//2
            step[dim] = patch_shape[dim] - overlap
    stop = start + n_patches * step
    
    patches = get_set_of_patch_indices(start, stop, step)
    # add the centeric cube:
    patches = np.vstack((patches, (img_shape - patch_shape)//2))
    
    return patches

def patching(img_shape, patch_shape, overlap = None):
    '''
    Patching for each image.
    img_shape: numpy.ndarray or tuple; shape = (3,)
    patch_shape: numpy.ndarray or tuple; shape = (3,)
    overlap: int or tuple or numpy.ndarray; 
             shape = (3,); 
             If None, only take the autofit patching strategy, 
                  otherwise symmetrically cover the image with patches as much as possible.
                  This is for the augmentation consideration to verify the diversity of input samples.
                  It may not be compulsary.
    Return list of bottom left corner coordinate of patches.
    '''
#     pdb.set_trace()
    img_shape = np.asarray(img_shape)
    patch_shape = np.asarray(patch_shape)
    
    patches = _patching_autofit(img_shape, patch_shape)
    if overlap is None:
        return patches
    
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(img_shape))
    else:
        overlap = np.asarray(overlap)
    n_patches = np.ceil(img_shape / (patch_shape - overlap))
    overflow = patch_shape * n_patches - (n_patches - 1) * overlap - img_shape
    start = -overflow//2
    step = patch_shape - overlap
    stop = start + n_patches * step
    
    patches = np.vstack((patches,get_set_of_patch_indices(start, stop, step)))
    
    return patches

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def create_id_index_patch_list(id_index_list, data_file, patch_shape, patch_overlap=None, trivial=True):
    '''
    id_index_list: id_index is the index of .h5.keys()
    data_file: .h5 file path
    patch_shape: numpy.ndarray or tuple; shape = (3,)
    patch_overlap: overlap among patches
    trivial: If True, use tqdm.
    Return: list of (index of .h5.keys(), bottom left corner coordinates of one patch)
    '''
    id_index_patch_list = []
    with h5py.File(data_file,'r') as h5_file:
        id_list = list(h5_file.keys())
        for index in tqdm(id_index_list,desc = 'Creating (id_index, patch_corner) list') if trivial else id_index_list:
            brain_width = h5_file[id_list[index]]['brain_width']
            brain_wide_img_shape = brain_width[1] - brain_width[0] + 1
            patches = patching(brain_wide_img_shape, patch_shape, overlap = patch_overlap)
            id_index_patch_list.extend(itertools.product([index], patches))
    return id_index_patch_list


def get_patch_from_3d_data(data, patch_shape, patch_corner):
    """
    Returns a 4D patch from a 4D image.
    data: 4D image shape=(4,_,_,_)
    patch_shape: numpy.ndarray or tuple; shape = (3,)
    patch_corner: bottom left corner coordinates of one patch.
    return: shape=(4,_,_,_)
    """
    patch_corner = np.asarray(patch_corner, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    img_shape = data.shape[-3:]
    if np.any(patch_corner < 0) or np.any((patch_corner + patch_shape) > img_shape):
        data, patch_corner = fix_out_of_bound_patch_attempt(data, patch_shape, patch_corner)
    return data[:, 
                patch_corner[0]:patch_corner[0]+patch_shape[0],
                patch_corner[1]:patch_corner[1]+patch_shape[1],
                patch_corner[2]:patch_corner[2]+patch_shape[2]]

def get_data_from_file(data_file, id_index_patch, patch_shape):
        '''
        Load image patch from .h5 file and mix 4 modalities into one 4d ndarray. 
        data_file: .h5 file of datasets.
        id_index_patch: tuple, (id_index is the index of .h5.keys(), 
                                patch is the patch corner coordinate).
        patch_shape: numpy.ndarray or tuple; shape = (3,)
        Return x.shape = (4,_,_,_); 
               y.shape = (1,_,_,_) for training dataset with seg.nii.gz; 
               y = 0 for validation and test datasets
        '''
    #     pdb.set_trace()
        id_index, corner = id_index_patch

        with h5py.File(data_file,'r') as h5_file:
            sub_id = list(h5_file.keys())[id_index]
            brain_width = h5_file[sub_id]['brain_width']

            data = []
            truth = []
            for name, img in h5_file[sub_id].items():
                if name == 'brain_width':
                    continue
                brain_wise_img = img[brain_width[0,0]:brain_width[1,0]+1,
                                     brain_width[0,1]:brain_width[1,1]+1,
                                     brain_width[0,2]:brain_width[1,2]+1]
                if name.split('_')[-1].split('.')[0] == 'seg':
                    truth.append(brain_wise_img)
                else:
                    data.append(brain_wise_img)

        x = get_patch_from_3d_data(np.asarray(data), patch_shape, corner)
        y = get_patch_from_3d_data(np.asarray(truth), patch_shape, corner) if truth else None
        return x, y


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_corner, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    data: 4D image shape=(4,_,_,_)
    patch_shape: shape=(3,)
    patch_corner: bottom left corner coordinates of one patch.
    """
    img_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_corner < 0) * patch_corner)
    pad_after = np.abs(((patch_corner + patch_shape) > img_shape) * 
                       ((patch_corner + patch_shape) - img_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
#     data = np.pad(data, pad_args, mode="edge")
    data = np.pad(data, pad_args, 'constant',constant_values=0)
    patch_corner += pad_before
    return data, patch_corner


def stitch(patch_list, patch_corners, data_shape):
    '''
    To put patches together.
    Overlapped places would take the mean value.
    patch_list: one list of ndarray patches, patch_shape=(3,_,_,_)
    patch_corners: bottom left corner coordinates of patches.
    data_shape: brain-wise shape after stitching, shape=(3,_,_,_)
    Return the brain-wised predicted ndarray shape=data_shape 
    '''
#     pdb.set_trace()
    data = np.zeros(data_shape)
    img_shape = data_shape[-3:]
    count = np.zeros(data_shape)
    patch_shape = patch_list[0].shape[-3:]
    for patch, corner in zip(patch_list, patch_corners):
        if np.any(corner < 0):
            start_edge = np.asarray((corner < 0) * np.abs(corner), dtype=np.int)
            patch = patch[:, start_edge[0]:, start_edge[1]:, start_edge[2]:]
            corner[corner < 0] = 0
        if np.any((corner + patch_shape) >= img_shape):
            end_edge = np.asarray(patch_shape - (((corner + patch_shape) >= img_shape) 
                                                 *((corner + patch_shape) - img_shape)), 
                                  dtype=np.int)
            patch = patch[:, :end_edge[0], :end_edge[1], :end_edge[2]]
        patch_index = np.zeros(data_shape, dtype=np.bool)
        patch_index[:,
                    corner[0]:corner[0]+patch.shape[-3],
                    corner[1]:corner[1]+patch.shape[-2],
                    corner[2]:corner[2]+patch.shape[-1]] = True
        data[patch_index] += patch.flatten()
        count[patch_index] += 1
    if np.any(count==0):
        print_red('Some empty place during stitching!')
        count[count==0] = 1
    
    return data/count