import pdb
import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
import generator
from loss import WeightedDiceLoss
from helper import calc_param_size, print_red
from nas import ShellNet
import sys
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle

DEBUG_FLAG = True

class Searching():
    '''
    Main class for searching
    '''
    def __init__(self, jupyter = True):
        self.jupyter = jupyter
        self._init_config()
        self._init_log()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self.check_resume()
    
    def _init_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',type=str,default='config.yml',
                            help='Configuration file to use')
        if self.jupyter: # for jupyter notebook
            self.args = parser.parse_args(args=[])
        else:  # for shell
            self.args = parser.parse_args()
        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        return
    
    def _init_log(self):
        self.log_dir = self.config['search']['log_dir']
#         self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboardX_log'))
        
    def _init_device(self):
        if self.config['search']['gpu'] and torch.cuda.is_available() :
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        else:
            print_red('No gpu devices available!, we will use cpu')
            self.device = torch.device('cpu')
        return
    
    def _init_dataset(self):
        dataset = generator.Dataset()
        self.train_generator = dataset.train_generator
        self.val_generator = dataset.val_generator
        return
    
    def _init_model(self):
        self.model = ShellNet(in_channels=len(self.config['data']['all_mods']), 
                              init_n_kernels=self.config['search']['init_n_kernels'], 
                              out_channels=len(self.config['data']['labels']), 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              normal_w_share=self.config['search']['normal_w_share'], 
                              channel_change=self.config['search']['channel_change']).to(self.device)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.loss = WeightedDiceLoss().to(self.device)

        self.optim_shell = Adam(self.model.alphas(), lr=3e-4)
        self.optim_kernel = AdaBound(self.model.kernel.parameters(), lr=1e-3, weight_decay=5e-4)
        self.kernel_lr_scheduler = CosineAnnealingLR(self.optim_kernel, 
                                                     self.config['search']['epochs'], eta_min=1e-3)
        

    def check_resume(self):
        self.save_path = self.config['search']['last_save']
        if os.path.exists(self.save_path):
            state_dicts = torch.load(self.save_path, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.geno_count = state_dicts['geno_count']
            self.history = state_dicts['history']
            self.model.load_state_dict(state_dicts['model_param'])
            self.optim_shell.load_state_dict(state_dicts['optim_shell'])
            self.optim_kernel.load_state_dict(state_dicts['optim_kernel'])
            self.kernel_lr_scheduler.load_state_dict(state_dicts['scheduler'])
        else:
            self.epoch = 0
            self.geno_count = Counter()
            self.history = defaultdict(list)

    def search(self):
        '''
        Return the best genotype
        '''
        pdb.set_trace()
        geno_file = self.config['search']['geno_file']
        if os.path.exists(geno_file):
            print('{} exists.'.format(geno_file))
            with open(geno_file, 'rb') as f:
                return pickle.load(f)

        best_genotype = None
        best_geno_count = self.config['search']['best_geno_count']
        n_epochs = self.config['search']['epochs']
        for epoch in range(n_epochs):
            genotype = self.model.genotype()
            self.geno_count[str(genotype)] += 1
            if self.geno_count[str(genotype)] >= best_geno_count:
                print('>= best_geno_count: ({})'.format(best_geno_count))
                best_genotype = genotype
                break

            shell_loss, kernel_loss = self.train()
            val_loss = self.validate()
            self.kernel_lr_scheduler.step()
            self.history['shell_loss'].append(shell_loss)
            self.history['kernel_loss'].append(kernel_loss)
            self.history['val_loss'].append(val_loss)
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'geno_count': self.geno_count,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim_shell': self.optim_shell.state_dict(),
                'optim_kernel': self.optim_kernel.state_dict(),
                'scheduler': self.kernel_lr_scheduler.state_dict()
            }
            torch.save(state_dicts, self.save_path)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch > 2:
                break
                
        if best_genotype is None:
            best_genotype = self.geno_count.most_common(1)
        with open(geno_file, 'wb') as f:
            pickle.dump(best_genotype, f)
        return best_genotype
        
    
    def train(self):
        '''
        Searching | Training process
        To do optim_shell.step() and optim_kernel.step() in turn.
        '''
        self.model.train()
        train_epoch = self.train_generator.epoch()
        val_epoch = self.val_generator.epoch()
        n_steps = self.train_generator.steps_per_epoch
        sum_loss = 0
        sum_val_loss = 0
        with tqdm(train_epoch, total = n_steps,
                  desc = 'Searching | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                try:
                    val_x, val_y_truth = next(val_epoch)
                except StopIteration:
                    val_epoch = self.val_generator.epoch()
                    val_x, val_y_truth = next(val_epoch)
                val_x = torch.as_tensor(val_x, device=self.device, dtype=torch.float)
                val_y_truth = torch.as_tensor(val_y_truth, device=self.device, dtype=torch.float)

                # optim_shell
                self.optim_shell.zero_grad()
                val_y_pred = self.model(val_x)
                val_loss = self.loss(val_y_pred, val_y_truth)
                sum_val_loss += val_loss.item()
                val_loss.backward()
                self.optim_shell.step()
                
                # optim_kernel
                self.optim_kernel.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.kernel.parameters(),
                                         self.config['search']['grad_clip'])
                self.optim_kernel.step()
                
                # postfix for progress bar
                postfix = OrderedDict()
                postfix['Loss(optim_shell)'] = round(sum_val_loss/(step+1), 3)
                postfix['Loss(optim_kernel)'] = round(sum_loss/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if DEBUG_FLAG and step > 3:
                    break
                
        return round(sum_val_loss/n_steps, 3), round(sum_loss/n_steps, 3)
    
    
    def validate(self):
        '''
        Searching | Validation process
        '''
        self.model.eval()
        n_steps = self.val_generator.steps_per_epoch
        sum_loss = 0
        with tqdm(self.val_generator.epoch(), total = n_steps,
                  desc = 'Searching | Epoch {} | Val'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                pbar.set_postfix(Loss=round(sum_loss/(step+1), 3))
                
                if DEBUG_FLAG and step > 3:
                    break
        return round(sum_loss/n_steps, 3)

    
if __name__ == '__main__':
    search = Searching(jupyter = False)
    genotype = search.search()