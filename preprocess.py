import h5py
import nibabel as nib
import os
import glob
from dev_tools.my_tools import print_red, minmax_normalize
import pdb
import numpy as np
import yaml
from tqdm.notebook import tqdm
import pickle
import time
from random import shuffle


def create_h5(source_folder, overwrite=False, config_yml='config.yml'):
    '''
    From the downloaded unziped folder to normalized .h5 file.
    Return .h5 path.
    '''
    with open(config_yml) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        
    try:
        affine = np.load(config['data']['affine_file'])
    except FileNotFoundError:
        affine = None
    
    dataset_type = source_folder.split('_')[-1].lower() # 'training' or 'validation' or 'testing'
    target = os.path.join('data',dataset_type + '.h5')
    
    if os.path.exists(target) and not overwrite:
        print('{:s} exists already.'.format(target))
        return target
    
    with open(config['data']['mean_std_file'],'rb') as f:
        mean_std_values = pickle.load(f)
    
    with h5py.File(target,'w') as h5_file:
        img_dirs  = glob.glob(os.path.join(source_folder,'*/*' 
                                             if dataset_type == 'training' else '*'))
        # for each subject:
        for img_dir in tqdm(img_dirs,desc='writing {:s}'.format(target)):
            if not os.path.isdir(img_dir):
                continue
            sub_id = img_dir.split('/')[-1]
            h5_subid = h5_file.create_group(sub_id)
            brain_widths = []
            # different modalities:
            for mod_file in os.listdir(img_dir): 
                img = nib.load(os.path.join(img_dir,mod_file))
                if affine is None:
                    affine = img.affine
                    np.save(config['data']['affine_file'],affine)
                img_npy = img.get_data()
                mod = mod_file.split('_')[-1].split('.')[0]
                if mod != 'seg':
                    img_npy = normalize(img_npy,
                                        mean = mean_std_values['{:s}_mean'.format(mod)],
                                        std = mean_std_values['{:s}_std'.format(mod)])
                    brain_widths.append(cal_outline(img_npy))
                h5_subid.create_dataset(mod_file,data=img_npy)
            start_edge = np.min(brain_widths,axis=0)[0]
            end_edge = np.max(brain_widths,axis=0)[1]
            brain_width = np.vstack((start_edge,end_edge))
            h5_subid.create_dataset('brain_width',data=brain_width)
        num_subs = len(h5_file)
        
    # update config.yml
    with open(config_yml,'w') as f:
        config['data'].update({'{:s}_h5'.format(dataset_type):target,
                               'len_{:s}'.format(dataset_type):num_subs})
        yaml.dump(config,f)
        
    return target

def cal_outline(img_npy):
    '''
    Return an numpy array shape=(2,3), indicating the outline of the 3D brain area.
    '''
    brain_index = np.asarray(np.nonzero(img_npy))
    start_edge = np.maximum(np.min(brain_index,axis=1)-1,0)
    end_edge = np.minimum(np.max(brain_index,axis=1)+1,img_npy.shape)
    
    return np.vstack((start_edge,end_edge))

def normalize(img_npy,mean,std,offset=0.1, mul_factor=100):
    '''
    Offset and mul_factor are used to make a distinction between brain voxel and background(zeros).
    '''
    brain_index = np.nonzero(img_npy)
    img_npy[brain_index] = (minmax_normalize((img_npy[brain_index]-mean)/std) + offset) * mul_factor
    return img_npy


def cal_mean_std(source_folder, overwrite=False,config_yml = 'config.yml'):
    '''
    We only care about non-zero voxels which are voxels in brain areas.
    This function calcultes the mean value and standard deviation of all non-zero voxels for each modalities.
    Return a dictionary {'t1_mean': t1 mean value,'t1_std': t1 std value,'t2_mean': ...,'t2_std': ..., ...}
    '''
    with open(config_yml) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        saved_path = config['data']['mean_std_file']
    
    if os.path.exists(saved_path) and not overwrite:
        print('{:s} exists already.'.format(saved_path))
        return
    
    sub_dirs = glob.glob(os.path.join(source_folder,'*/*')) # Specific Design
    
    mean_std_values = {}
    
    for mod in config['data']['all_mods']:
        mean = 0
        amount = 0
        for sub_dir in tqdm(sub_dirs,
                             desc='Calculating {:s}\'s mean value'
                             .format(mod)):
            file_name = os.path.join(sub_dir,sub_dir.split('/')[-1]+'_{:s}.nii.gz'.format(mod))
            img_npy = nib.load(file_name).get_data()
            brain_area = img_npy[np.nonzero(img_npy)]
            mean += np.sum(brain_area)
            amount += len(brain_area)
        mean /= amount
        mean_std_values['{:s}_mean'.format(mod)] = round(mean,4)
        print('{:s}\'s mean value = {:.2f}'.format(mod,mean))
        
        std = 0
        for sub_dir in tqdm(sub_dirs,
                             desc='Calculating {:s}\'s std value'
                             .format(mod)):
            file_name = os.path.join(sub_dir,sub_dir.split('/')[-1]+'_{:s}.nii.gz'.format(mod))
            img_npy = nib.load(file_name).get_data()
            brain_area = img_npy[np.nonzero(img_npy)]
            std += np.sum((brain_area-mean)**2)
        std = np.sqrt(std/amount)
        mean_std_values['{:s}_std'.format(mod)] = round(std,4)
        print('{:s}\'s std value = {:.2f}'.format(mod,std))
    print(mean_std_values)
    
    with open(saved_path,'wb') as f:
        pickle.dump(mean_std_values,f)
    return
    
def cross_val_split(num_sbjs, saved_path, num_folds=5, overwrite=False):
    '''
    To generate num_folds cross validation.
    Return {'train_list_0':[],'val_list_0':[],...}
    '''
    if os.path.exists(saved_path) and not overwrite:
        print('{:s} exists already.'.format(saved_path))
        return
    subid_indices = list(range(num_sbjs))
    shuffle(subid_indices)
    res = {}
    for i in range(num_folds):
        left = int(i/num_folds * num_sbjs)
        right = int((i+1)/num_folds * num_sbjs)
        res['train_list_{:d}'.format(i)] = subid_indices[:left] + subid_indices[right:]
        res['val_list_{:d}'.format(i)] = subid_indices[left : right]
    for i in res.values():
        shuffle(i)
    with open(saved_path,'wb') as f:
        pickle.dump(res,f)
    return
   
def preprocess(config_yml='config.yml'):
    '''
    From downloaded unziped folders to Training.h5 Validation.h5 and Testing.h5 
    '''
    with open(config_yml) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    cal_mean_std(source_folder=config['data']['source_train'])

    create_h5(config['data']['source_train'])
    create_h5(config['data']['source_val'])
    create_h5(config['data']['source_test'])   
    
    # split for cross validation
    cross_val_file = config['data']['cross_val_indices']
    cross_val_split(config['data']['len_training'], cross_val_file)
    
    return

if __name__ == '__main__':
    from tqdm import tqdm
    preprocess()

                          