import pdb
import os
import torch
import torch.nn as nn
import generator
from loss import WeightedDiceLoss
from helper import calc_param_size
from searched import SearchedNet
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
from genotype import Genotype
import shutil
from search import Base

DEBUG_FLAG = False

    
class Training(Base):
    '''
    Training the searched network
    '''
    def __init__(self, jupyter=True):
        super().__init__(jupyter=jupyter)
        self._init_model()
        self.check_resume()
    
    def _init_model(self):
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
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.loss = WeightedDiceLoss().to(self.device)

        self.optim = Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optim,factor=0.5)
#         self.optim = AdaBound(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
#         self.lr_scheduler = CosineAnnealingLR(self.optim, 
#                                               self.config['train']['epochs'], eta_min=1e-3)
        

    def check_resume(self):
        self.last_save = self.config['train']['last_save']
        self.best_shot = self.config['train']['best_shot']
        if os.path.exists(self.last_save):
            state_dicts = torch.load(self.last_save, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            self.model.load_state_dict(state_dicts['model_param'])
            self.optim.load_state_dict(state_dicts['optim'])
            self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_val_loss = 1.0

    def main_run(self):
        '''
        Return the best genotype in tuple:
        (best_genotype: str(Genotype), geno_count: int)
        '''
#         pdb.set_trace()

        n_epochs = self.config['train']['epochs']
        
        for epoch in range(n_epochs):
            is_best = False
            loss = self.train()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            if val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss
            }
            torch.save(state_dicts, self.last_save)
            
            if is_best:
                shutil.copy(self.last_save, self.best_shot)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 1:
                break
        print('Training Finished.')
        return 
        
    
    def train(self):
        '''
        Training | Training process
        '''
        self.model.train()
        n_steps = self.train_generator.steps_per_epoch
        sum_loss = 0
        with tqdm(self.train_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)

                self.optim.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(),
#                                          self.config['search']['grad_clip'])
                self.optim.step()
                
                pbar.set_postfix(Loss=round(sum_loss/(step+1), 3))
                
                if DEBUG_FLAG and step > 1:
                    break
                
        return round(sum_loss/n_steps, 3)
    
    
    def validate(self):
        '''
        Training | Validation process
        '''
        self.model.eval()
        n_steps = self.val_generator.steps_per_epoch
        sum_loss = 0
        with tqdm(self.val_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Val'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                pbar.set_postfix(Loss=round(sum_loss/(step+1), 3))
                
                if DEBUG_FLAG and step > 1:
                    break
        return round(sum_loss/n_steps, 3)

    
if __name__ == '__main__':
    training = Training(jupyter = False)
    training.main_run()