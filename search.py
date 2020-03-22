import pdb
import argparse
import yaml
from tensorboardX import SummaryWriter
import os
import time
import torch
import torch.nn as nn
import generator
from loss import WeightedDiceLoss
from helper import calc_param_size, print_red, visualize
from nas import ShellNet#, #ShellConsole
import sys
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
from tqdm import tqdm

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
        # Setup loss function
        self.loss = WeightedDiceLoss().to(self.device)
        # Setup Model
        self.model = ShellNet(in_channels=len(self.config['data']['all_mods']), 
                              init_n_kernels=self.config['search']['init_n_kernels'], 
                              out_channels=len(self.config['data']['labels']), 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              normal_w_share=self.config['search']['normal_w_share'], 
                              channel_change=self.config['search']['channel_change']).to(self.device)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))

        self.optim_shell = Adam(self.model.alphas(), lr=3e-4)
        self.optim_kernel = AdaBound(self.model.kernel.parameters(), lr=1e-3, weight_decay=5e-4)
        self.kernel_lr_scheduler = CosineAnnealingLR(self.optim_kernel, self.config['search']['epochs'], eta_min=1e-3)
        
#         self.shell_console = ShellConsole(self.model, self.optim_shell, self.loss)
        
        
#         pdb.set_trace()
#         x = torch.randn(1, 4, 64, 64, 64)
#         x = torch.as_tensor(x, device=torch.device('cuda'))
        
#         y = self.model(x)
#         pdb.set_trace()

    def check_resume(self):
        self.epoch = 0
        self.geno_count = Counter()

    def search(self):
        pdb.set_trace()
        print('Searching starts:')
        start_time = time.time()
        for epoch in range(self.config['search']['epochs']):
            self.epoch += epoch
            # genotype:
            genotype = self.model.genotype()
            self.geno_count[str(genotype)] += 1
            if self.geno_count[str(genotype)] >= self.config['search']['best_geno_count']:
                self.genotype = genotype
                break
                
#             visualize(genotype.down, 
#                       os.path.join(self.log_dir, 'epoch_{}_db'.format(epoch)), 
#                       'Downward Block')
#             visualize(genotype.up, 
#                       os.path.join(self.log_dir, 'epoch_{}_ub'.format(epoch)), 
#                       'Upward Block')
            self.train()
            
            # self.validate()
            
            # criteria logs
            
            # save & print logs
            
            
            
            self.kernel_lr_scheduler.step()
        self.genotype = self.geno_count.most_common(1)
        # print logs
    
    def train(self):
        self.model.train()
        train_epoch = self.train_generator.epoch()
        val_epoch = self.val_generator.epoch()
        with tqdm(self.train_generator.epoch(), 
                  desc = 'Searching | Training | Epoch {}'.format(self.epoch),
                  total = self.train_generator.steps_per_epoch) as pbar:
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
                val_loss.backward()
                self.optim_shell.step()
                
                # optim_kernel
                self.optim_kernel.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.kernel.parameters(),
                                         self.config['search']['grad_clip'])
                self.optim_kernel.step()
                
            
                pbar.set_postfix(ShellLoss = '{:.3f}'.format(val_loss.item()), 
                                 KernelLoss = '{:.3f}'.format(loss.item()))
                if val_loss.item() > 1 or loss.item() > 1 or val_loss.item() < 0 or loss.item() < 0:
                    pdb.set_trace()
            
        return
    
    
    def validate(self):
        return

    
if __name__ == '__main__':
    search_network = Searching(jupyter = False)
#     search_network.run()