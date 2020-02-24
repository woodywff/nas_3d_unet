from random import shuffle
import yaml
import numpy as np
import h5py
from patches import create_id_index_patch_list, get_patch_from_3d_data
from augment import do_augment, random_permutation_x_y
import pickle
from tqdm.notebook import tqdm
import os
import pdb

class Dataset():
    '''
    BraTS dataset pipeline for training and validation process.
    It provides to generators for training and validation respectively.
    '''
    def __init__(self, config_yml = 'config.yml', for_final_training=False):
        with open(config_yml) as f:
            self.data_config = yaml.load(f,Loader=yaml.FullLoader)['data']
        with open(self.data_config['cross_val_indices'],'rb') as f:
            self.cross_val_indices = pickle.load(f)
        self.for_final_training = for_final_training
            
    @property
    def _train_indices(self):
        return self.cross_val_indices['train_list_0'] + (self._val_indices if self.for_final_training else [])

    @property
    def _val_indices(self):
        return self.cross_val_indices['val_list_0']

    @property
    def train_generator(self):
        return Generator(self._train_indices, 
                         self.data_config['training_h5'], 
                         patch_shape = self.data_config['patch_shape'], 
                         patch_overlap = self.data_config['patch_overlap'],
                         batch_size= self.data_config['batch_size_train'], 
                         epochs = self.data_config['epochs'],
                         labels = self.data_config['labels'], 
                         augment = self.data_config['augment'], 
                         augment_flip = self.data_config['augment_flip'], 
                         augment_distortion_factor = self.data_config['augment_distortion_factor'], 
                         permute = self.data_config['permute'],
                         affine_file = self.data_config['affine_file'],
                         spe_file = self.data_config['spe_file'])
    @property
    def val_generator(self):
        return Generator(self._val_indices, 
                         self.data_config['training_h5'], 
                         patch_shape = self.data_config['patch_shape'], 
                         batch_size= self.data_config['batch_size_val'],
                         labels = self.data_config['labels'], 
                         shuffle_index_list = False,
                         spe_file = self.data_config['spe_file'])

class Generator():
    def __init__(self, indices_list, data_file, 
                       patch_shape, patch_overlap = None, 
                       batch_size=1, epochs=1, labels=None, 
                       augment=False, augment_flip=True, augment_distortion_factor=0.25, permute=False,
                       shuffle_index_list=True, 
                       affine_file = None, spe_file = None,
                       skip_health = True):
        self.indices_list = indices_list # list of indices in .h5.keys()
        self.data_file = data_file # .h5 file path
        self.patch_shape = [patch_shape] * 3 if isinstance(patch_shape,int) else patch_shape
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        self.epochs = epochs
        self.labels = labels
        self.augment = augment
        self.augment_flip = augment_flip
        self.augment_distortion_factor = augment_distortion_factor
        self.permute = permute # rotate and flip
        self.shuffle_index_list = shuffle_index_list
        self.affine_file = affine_file # affine.npy path
        self.spe_file = spe_file # steps per epoch .pkl
        self.skip_health = skip_health # True: skip none tumor images
        
        self.epoch_init()
        
    
    def epoch_init(self):
        '''
        The self.steps_per_epoch is needed by the tqdm outside the Generator class.
        So we need to calc it before each invoking of self.epoch().
        self.spe_file saves a dict, for each key named as spe_name we hold the corresponding spe value.
        For a certain Generator(), the spe value mainly changes when self.patch_overlap is not None or 0,
        that's when the overlap varies as a random int from 0 to self.patch_overlap (inclusive).
        '''
        patch_overlap = self.patch_overlap if not self.patch_overlap else random.randint(0,self.patch_overlap + 1)
        self.id_index_patch_list = create_id_index_patch_list(self.indices_list, 
                                                              self.data_file, 
                                                              self.patch_shape, 
                                                              patch_overlap)
        
#         pdb.set_trace()
        if not os.path.exists(self.spe_file):
            with open(self.spe_file, 'wb') as f:
                pickle.dump({},f)
        with open(self.spe_file, 'rb') as f:
            spes = pickle.load(f)
        spe_name = '{}_{}_{}_{}'.format(self.indices_list[0], self.patch_shape[0], patch_overlap, self.batch_size)
        if spes.get(spe_name) is None:
            self.steps_per_epoch = self.get_steps_per_epoch()
            spes[spe_name] = self.steps_per_epoch
            with open(self.spe_file, 'wb') as f:
                pickle.dump(spes,f)
        else:
            self.steps_per_epoch = spes[spe_name]
    
    def _get_num_patches(self):
        id_index_patch_list = self.id_index_patch_list.copy()
        count = 0
        for id_index_patch in tqdm(id_index_patch_list, desc = 'Calculating the number of patches'):
            x_list = []
            y_list = []
            self.add_data(x_list, y_list, id_index_patch, _augment = False, _permute = False)
            if len(x_list) > 0:
                count += 1
        return count
    
    def get_steps_per_epoch(self):
        return int(np.ceil(self._get_num_patches()/self.batch_size))
        
    def epoch(self):
        '''
        A generator for one epoch.
        If self.patch_overlap is set, for each epoch the paching results may be different, 
        so we need to self.epoch_init() each time.
        '''
        x_list = []
        y_list = []
            
        id_index_patch_list = self.id_index_patch_list.copy()
        if self.shuffle_index_list:
            shuffle(id_index_patch_list)
        while len(id_index_patch_list) > 0:
            id_index_patch = id_index_patch_list.pop()
            self.add_data(x_list, y_list, id_index_patch)
            if len(x_list) == self.batch_size or (len(id_index_patch_list) == 0 and len(x_list) > 0):
                yield self.convert_data(x_list, y_list)
#                 convert_data()
                x_list = []
                y_list = []
#         pdb.set_trace()
        if self.patch_overlap:
            self.epoch_init()
        return    
    
    def add_data(self, x_list, y_list, id_index_patch, _augment = True, _permute = True):
        '''
        Add qualified x,y to the generator list
        '''
    #     pdb.set_trace()
        # data.shape = (4,_,_,_), truth.shape = (1,_,_,_):
        data, truth = self.get_data_from_file(id_index_patch)
        # skip empty images
        if np.all(data == 0):
            return
        # skip none tumor images
        if self.skip_health and np.all(truth==0):
            return
        if self.augment and _augment:
            affine = np.load(self.affine_file)
            data, truth = do_augment(data, truth, affine, flip=self.augment_flip, 
                                     scale_deviation=self.augment_distortion_factor)
        if self.permute and _permute:
            assert data.shape[-1] == data.shape[-2] == data.shape[-3], 'Not a cubic patch!'
            data, truth = random_permutation_x_y(data, truth)
        x_list.append(data)
        y_list.append(truth)
        return
    
    
    def get_data_from_file(self, id_index_patch):
        '''
        Load image patch from .h5 file and mix 4 modalities into one 4d ndarray. 
        Return x.shape = (4,_,_,_); y.shape = (1,_,_,_)
        '''
    #     pdb.set_trace()
        id_index, patch = id_index_patch

        with h5py.File(self.data_file,'r') as h5_file:
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

        x = get_patch_from_3d_data(data, self.patch_shape, patch)
        y = get_patch_from_3d_data(truth, self.patch_shape, patch)
        return x, y

    def convert_data(self, x_list, y_list):
    #     pdb.set_trace()
        x = np.asarray(x_list)
        y = np.asarray(y_list)
        y = self.get_multi_class_labels(y)
        return x, y


    def get_multi_class_labels(self, truth):
        '''
        truth.shape is (batch_size,1,patch_shape[0],patch_shape[1],patch_shape[2])
        y.shape is (batch_size,len(labels),_,_,_)
        truth values:
            4: ET
            1+4: TC
            1+2+4: WT
        '''
        n_labels = len(self.labels)
        new_shape = [truth.shape[0], n_labels] + list(truth.shape[2:])
        y = np.zeros(new_shape, np.int8)

        y[:,0][np.logical_or(truth[:,0] == 1,truth[:,0] == 4)] = 1    #1
        y[:,1][np.logical_or(truth[:,0] == 1,truth[:,0] == 2, truth[:,0] == 4)] = 1 #2
        y[:,2][truth[:,0] == 4] = 1    #4
        return y
