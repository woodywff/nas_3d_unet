import os
import nibabel as nib
import numpy as np
import h5py
import pdb
# from tqdm import tqdm
from tqdm.notebook import tqdm
import time
from helper import print_red
import pickle
from genotype import Genotype
from searched import SearchedNet
from search import Base
from helper import calc_param_size
import torch
from patches import create_id_index_patch_list, get_data_from_file, stitch
    
class Prediction(Base):
    '''
    Prediction process
    '''
    def __init__(self, jupyter=True):
        super().__init__(jupyter=jupyter)
        self._init_model()
    
    def _init_dataset(self):
        '''
        This is an overridden function in super class (Base).
        '''
        self.img_shape = self.config['data']['img_shape']
        self.output_folder = self.config['predict']['output_folder']
        patch_shape = self.config['train']['patch_shape']
        self.patch_shape = [patch_shape] * 3 if isinstance(patch_shape,int) else patch_shape
        self.n_labels = len(self.config['data']['labels'])
        self.affine = np.load(self.config['data']['affine_file'])
        try:
            os.mkdir(self.output_folder)
        except FileExistsError:
            pass
    
    def _init_model(self):
        '''
        Load the best_shot trained model.
        '''
        geno_file = self.config['search']['geno_file']
        with open(geno_file, 'rb') as f:
            gene = eval(pickle.load(f)[0])
        self.model = SearchedNet(in_channels=len(self.config['data']['all_mods']), 
                              init_n_kernels=self.config['search']['init_n_kernels'], 
                              out_channels=len(self.config['data']['labels']), 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              channel_change=self.config['search']['channel_change'],
                              gene=gene).to(self.device)
        state_dicts = torch.load(self.config['train']['best_shot'], map_location=self.device)
        self.model.load_state_dict(state_dicts['model_param'])
        self.model.eval()
        
    def predict(self, h5file=None):
        '''
        Prediction on the h5file dataset.
        skull_mask is to avoid any predicted tumor voxel staying outside of the skull.
        '''
        if h5file is None:
            h5file = self.config['data']['validation_h5']
        target_folder = os.path.join(self.output_folder, h5file.split('/')[-1])
        try:
            os.mkdir(target_folder)
        except FileExistsError:
            print('{} exists already.'.format(target_folder))
            
        with h5py.File(h5file, 'r') as f:
            n_subs = len(f)
        for id_index in tqdm(range(n_subs), desc='Predicting images from {}'.format(h5file)):
            brain_width = None
            skull_mask = np.zeros(self.img_shape, dtype=np.uint8)
            with h5py.File(h5file, 'r') as f:
                sub_id = list(f.keys())[id_index]
                for name, value in f[sub_id].items():
                    if name == 'brain_width':
                        brain_width = np.asarray(value)
                        continue
                    skull_mask[np.nonzero(value)] = 1
            single_pred = self.single_img_predict(id_index, h5file, brain_width)
            tumor_pred = self.get_tumor_pred(single_pred) 
            tumor_pred *= skull_mask
            nib.Nifti1Image(tumor_pred, self.affine).to_filename(os.path.join(target_folder,
                                           '{}.nii.gz'.format(sub_id)))
        print('Prediction Finished.')
        return
            
    def single_img_predict(self, id_index, h5file, brain_width):
        '''
        id_index: the index of .h5.keys()
        h5file: .h5 file path
        brain_width: minimum cubic area that could encapsulate the brain.
        Return: original sized image in shape of (3,240,240,155)
        '''
        id_index_patch_list = create_id_index_patch_list([id_index], h5file, 
                                                         self.patch_shape, trivial=False)
        patch_pred_list = []
        for id_index_patch in id_index_patch_list:
            data, _ = get_data_from_file(h5file, id_index_patch, self.patch_shape)
            if np.all(data==0):
                patch_pred_list.append(np.zeros([self.n_label]+self.patch_shape))
                continue
            x = torch.as_tensor([data], device=self.device, dtype=torch.float)
            patch_pred_list.append(self.model(x)[0].detach().cpu().numpy())
        brain_wide_img_shape = [self.n_labels] + list(brain_width[1] - brain_width[0] + 1)
        brain_wide_pred = stitch(patch_pred_list, 
                                 list(zip(*id_index_patch_list))[1], 
                                 brain_wide_img_shape)
        final_shape = [self.n_labels] + self.img_shape
        final_pred = np.zeros(final_shape)
        final_pred[:,brain_width[0,0]:brain_width[1,0]+1,
                     brain_width[0,1]:brain_width[1,1]+1,
                     brain_width[0,2]:brain_width[1,2]+1] = brain_wide_pred
        return final_pred
        
    def get_tumor_pred(self, img_pred, threshold=0.5):
        '''
        From model's output values (last sigmoid layer's output) to one 3D iamge with tumor labels.
        img_pred: shape=(3,240,240,155)
        Return: tumor: ndarray with shape=(240,240,155)
        '''
        tumor = np.zeros(img_pred[0].shape, dtype=np.uint8)
        tumor[img_pred[1] >= threshold] = 2
        tumor[img_pred[0] >= threshold] = 1
        tumor[img_pred[2] >= threshold] = 4
        return tumor
        
if __name__ == '__main__':
    p = Prediction(jupyter = False)
