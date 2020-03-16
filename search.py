import pdb
import argparse
import yaml
from tensorboardX import SummaryWriter
import logging
import os
import time
import torch
import generator
from loss import WeightedDiceLoss
from helper import calc_param_size
from nas import ShellNet, ShellConsole
import sys
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import CosineAnnealingLR

class Searching():
    '''
    Main class for searching
    '''
    def __init__(self, jupyter = True):
        self.jupyter = jupyter
        self._init_configure()
        self._init_device()
        self._init_dataset()
        self._init_model()
        
        
    
    def _init_configure(self):
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
    
#     def _init_logger(self):
#         self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboardX_log'))
        
    def _init_device(self):
        if self.config['search']['gpu'] and torch.cuda.is_available() :
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        else:
            self.logger.warning('No gpu devices available!, we will use cpu')
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
        print('Param size = %.3f MB', calc_param_size(self.model))
        pdb.set_trace()
        self.optim_shell = Adam(self.model.alphas(), lr=3e-4)
        self.optim_kernel = AdaBound(self.model.kernel.parameters(), lr=1e-3, weight_decay=5e-4)
        self.kernel_lr_scheduler = CosineAnnealingLR(self.optim_kernel, self.config['search']['epochs'], eta_min=1e-3)
        
        self.shell_console = ShellConsole(self.model, self.optim_shell, self.loss)
        
        
        pdb.set_trace()
        x = torch.randn(1, 4, 64, 64, 64)
        x = torch.as_tensor(x, device=torch.device('cuda'))
        
        y = self.model(x)
        pdb.set_trace()
    def search(self):
        pass
    
    def train(self):
        pass
    
    def infer(self):
        pass
    

    


def test():
    for i in tqdm(range(10)):
        print(i)
    
if __name__ == '__main__':
    search_network = Searching(jupyter = False)
#     test()
#     search_network.run()