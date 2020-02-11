from random import shuffle
import yaml
import numpy as np
import h5py
from patches import create_id_index_patch_list, get_patch_from_3d_data
from augment import do_augment, random_permutation_x_y
import pickle

def data_generator(indices_list, data_file, 
                   patch_shape, patch_overlap = None, 
                   batch_size=1, labels=None, 
                   augment=False, augment_flip=True, augment_distortion_factor=0.25, permute=False,
                   shuffle_index_list=True, 
                   affine_file = None):
    '''
    Generator for training and validation datasets. 
    In this project training and val dataset both come from training.h5
    Patching = True
    Augmentation = True
    Overlap_label = True
    Pred_specific = True
    '''
#     pdb.set_trace()
    if isinstance(patch_shape,int):
        patch_shape = [patch_shape] * 3
    while True:
        x_list = []
        y_list = []
        id_index_patch_list = create_id_index_patch_list(indices_list, data_file, patch_shape, patch_overlap)
        if shuffle_index_list:
            shuffle(id_index_patch_list)
        while len(id_index_patch_list) > 0:
            id_index_patch = id_index_patch_list.pop()
            add_data(x_list, y_list, data_file, id_index_patch, patch_shape,
                     augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, 
                     permute=permute, affine_file = affine_file)
            if len(x_list) == batch_size or (len(id_index_patch_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list,labels=labels)
#                 convert_data(x_list, y_list,labels=labels)
                x_list = []
                y_list = []
    return

def add_data(x_list, y_list, data_file, id_index_patch, patch_shape,
             augment=False, augment_flip=False, augment_distortion_factor=0.25,
             permute=False, skip_health = True, affine_file = None):
    '''
    Add qualified x,y to the generator list
    '''
#     pdb.set_trace()
    # data.shape = (4,_,_,_), truth.shape = (1,_,_,_):
    data, truth = get_data_from_file(data_file, id_index_patch, patch_shape)
    # skip empty images
    if np.all(data == 0):
        return
    # skip none tumor images
    if skip_health and np.all(truth==0):
        return
    if augment:
        affine = np.load(affine_file)
        data, truth = do_augment(data, truth, affine, flip=augment_flip, 
                                 scale_deviation=augment_distortion_factor)
    if permute:
        assert data.shape[-1] == data.shape[-2] == data.shape[-3], 'Not a cubic patch!'
        data, truth = random_permutation_x_y(data, truth)
    x_list.append(data)
    y_list.append(truth)
    return


def get_data_from_file(data_file, id_index_patch, patch_shape):
    '''
    Load image patch from .h5 file and mix 4 modalities into one 4d ndarray. 
    Return x.shape = (4,_,_,_); y.shape = (1,_,_,_)
    '''
#     pdb.set_trace()
    id_index, patch = id_index_patch
    
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
    data = np.asarray(data)
    truth = np.asarray(truth)
    
    x = get_patch_from_3d_data(data, patch_shape, patch)
    y = get_patch_from_3d_data(truth, patch_shape, patch)
    return x, y

def convert_data(x_list, y_list, labels=None):
#     pdb.set_trace()
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    y = get_multi_class_labels(y, labels=labels)
    return x, y


def get_multi_class_labels(truth, labels=(1,2,4)):
    '''
    truth.shape is (batch_size,1,patch_shape[0],patch_shape[1],patch_shape[2])
    y.shape is (batch_size,len(labels),_,_,_)
    truth values:
        4: ET
        1+4: TC
        1+2+4: WT
    '''
    n_labels = len(labels)
    new_shape = [truth.shape[0], n_labels] + list(truth.shape[2:])
    y = np.zeros(new_shape, np.int8)
    
    y[:,0][np.logical_or(truth[:,0] == 1,truth[:,0] == 4)] = 1    #1
    y[:,1][np.logical_or(truth[:,0] == 1,truth[:,0] == 2, truth[:,0] == 4)] = 1 #2
    y[:,2][truth[:,0] == 4] = 1    #4
    return y


def get_training_and_validation_generators(config_yml='config.yml',for_final_training=False):
    '''
    for_final_training: if True, all subjects would be trained.
    '''
    with open(config_yml) as f:
        data_config = yaml.load(f,Loader=yaml.FullLoader)['data']
    
    # load indices list for training and validation
    with open(data_config['cross_val_indices'],'rb') as f:
        cross_val_indices = pickle.load(f)

    train_indices = cross_val_indices['train_list_0']
    val_indices = cross_val_indices['val_list_0']
    if for_final_training:
        train_indices += val_indices
    
    # generator for training and validation
    train_generator = data_generator(train_indices, 
                                        data_config['training_h5'], 
                                        patch_shape = data_config['patch_shape'], 
                                        patch_overlap = data_config['patch_overlap'],
                                        batch_size= data_config['batch_size_train'], 
                                        labels = data_config['labels'], 
                                        augment = data_config['augment'], 
                                        augment_flip = data_config['augment_flip'], 
                                        augment_distortion_factor = data_config['augment_distortion_factor'], 
                                        permute = data_config['permute'],
                                        affine_file = data_config['affine_file'])
    val_generator = data_generator(val_indices, 
                                   data_config['training_h5'], 
                                   patch_shape = data_config['patch_shape'], 
                                   patch_overlap = data_config['patch_overlap'],
                                   batch_size= data_config['batch_size_val'],
                                   labels = data_config['labels'], 
                                   shuffle_index_list = False)
    return train_generator, val_generator
