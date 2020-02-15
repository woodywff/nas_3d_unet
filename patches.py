import numpy as np
import pdb
from dev_tools.my_tools import print_red
import h5py
import itertools
from tqdm.notebook import tqdm

def _patching_autofit(image_shape, patch_shape):
    '''
    Autofit patching strategy:
        Symmetrically cover the image with patches without beyond boundary parts as far as possible.
    image_shape: numpy.ndarray; shape = (3,)
    patch_shape: numpy.ndarray; shape = (3,)
    '''
    n_dim = len(image_shape)
    n_patches = np.ceil(image_shape / patch_shape)
    start = np.zeros(n_dim)
    step = np.zeros(n_dim)
    for dim in range(n_dim):
        if n_patches[dim] == 1:
            start[dim] = -(patch_shape[dim] - image_shape[dim])//2
            step[dim] = patch_shape[dim]
        else:
            overlap = np.ceil(n_patches[dim] * patch_shape[dim] - image_shape[dim])/(n_patches[dim] - 1)
            overflow = n_patches[dim] * patch_shape[dim] - (n_patches[dim] - 1) * overlap - image_shape[dim]
            start[dim] = - overflow//2
            step[dim] = patch_shape[dim] - overlap
    stop = start + n_patches * step
    
    patches = get_set_of_patch_indices(start, stop, step)
    # add the centeric cube:
    patches = np.vstack((patches, (image_shape - patch_shape)//2))
    
    return patches

def patching(image_shape, patch_shape, overlap = None):
    '''
    Patching for each image.
    image_shape: numpy.ndarray or tuple; shape = (3,)
    patch_shape: numpy.ndarray or tuple; shape = (3,)
    overlap: int or tuple or numpy.ndarray; shape = (3,); If None, only take the autofit patching strategy, 
                  otherwise symmetrically cover the image with patches as much as possible.
                  This is for the augmentation consideration to verify the diversity of input samples.
                  It may not be compulsary.
    Return list of bottom left corner cords of patches.
    '''
#     pdb.set_trace()
    image_shape = np.asarray(image_shape)
    patch_shape = np.asarray(patch_shape)
    
    patches = _patching_autofit(image_shape, patch_shape)
    if overlap is None:
        return patches
    
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    else:
        overlap = np.asarray(overlap)
    n_patches = np.ceil(image_shape / (patch_shape - overlap))
    overflow = patch_shape * n_patches - (n_patches - 1) * overlap - image_shape
    start = -overflow//2
    step = patch_shape - overlap
    stop = start + n_patches * step
    
    patches = np.vstack((patches,get_set_of_patch_indices(start, stop, step)))
    
    return patches

# def patching_hardcode128(image_shape, patch_shape, center_patch=True, pdb_set=False):
# #     pdb.set_trace()
#     image_shape = np.asarray(image_shape)
#     patch_shape = np.asarray(patch_shape)
#     if pdb_set:
#         if np.any(np.array(2*np.array(patch_shape) - np.array(image_shape))<=0):
#             print_red('error patch: too large')
#         if  np.any(np.array(image_shape-patch_shape)<=0):
#             print_red('error patch: too small')
#     start_2 = np.asarray(image_shape - patch_shape)
#     start_2[start_2 < 0] = 0
#     patches = np.array([[0,         0,         0         ],
#                         [start_2[0],0,         0         ],
#                         [0,         start_2[1],0         ],
#                         [0,         0,         start_2[2]],
#                         [start_2[0],start_2[1],0         ],
#                         [start_2[0],start_2[1],start_2[2]],
#                         [start_2[0],0,         start_2[2]],
#                         [0,         start_2[1],start_2[2]]])
#     if center_patch:
#         patches = np.vstack((patches, (image_shape - patch_shape)//2))
#     return patches

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def create_id_index_patch_list(id_index_list, data_file, patch_shape, patch_overlap = None):
    '''
    id_index_list: id_index is the index of .h5.keys()
    data_file: .h5 file path
    patch_shape: shape = (3,)
    patch_overlap: overlap among patches
    Return: list of (subject id, bottom left corner coordinates of one patch)
    '''
    id_index_patch_list = []
    with h5py.File(data_file,'r') as h5_file:
        id_list = list(h5_file.keys())
        for index in tqdm(id_index_list,desc = 'Creating (id_index, patch) list'):
            brain_width = h5_file[id_list[index]]['brain_width']
            image_shape = brain_width[1] - brain_width[0] + 1
            patches = patching(image_shape, patch_shape, overlap = patch_overlap)
            id_index_patch_list.extend(itertools.product([index], patches))
    return id_index_patch_list


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
#     data = np.pad(data, pad_args, mode="edge")
    data = np.pad(data, pad_args, 'constant',constant_values=0)
    patch_index += pad_before
    return data, patch_index


# def reconstruct_from_patches(patches, patch_indices, data_shape, default_value=0):
#     """
#     Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
#     patches are averaged.
#     :param patches: List of numpy array patches.
#     :param patch_indices: List of indices that corresponds to the list of patches.
#     :param data_shape: Shape of the array from which the patches were extracted.
#     :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
#     be overwritten.
#     :return: numpy array containing the data reconstructed by the patches.
#     """
# #     pdb.set_trace()
#     data = np.ones(data_shape) * default_value
#     image_shape = data_shape[-3:]
#     count = np.zeros(data_shape, dtype=np.int)
#     image_patch_shape = patches[0].shape[-3:]
#     for patch, index in zip(patches, patch_indices):
#         if np.any(index < 0):
#             fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
#             patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
#             index[index < 0] = 0
#         if np.any((index + image_patch_shape) >= image_shape):
#             fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
#                                                         * ((index + image_patch_shape) - image_shape)), dtype=np.int)
#             patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
#         patch_index = np.zeros(data_shape, dtype=np.bool)
#         patch_index[...,
#                     index[0]:index[0]+patch.shape[-3],
#                     index[1]:index[1]+patch.shape[-2],
#                     index[2]:index[2]+patch.shape[-1]] = True
#         patch_data = np.zeros(data_shape)
#         patch_data[patch_index] = patch.flatten()

#         new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
#         data[new_data_index] = patch_data[new_data_index]

#         averaged_data_index = np.logical_and(patch_index, count > 0)
#         if np.any(averaged_data_index):
#             data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
#         count[patch_index] += 1
#     return data

