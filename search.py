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
from nas import NasShell

class Searching():
    '''
    Main class for searching
    '''
    def __init__(self, jupyter = True):
        self.jupyter = jupyter
        self._init_configure()
        self._init_logger()
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
    
    def _init_logger(self):
        log_dir = self.config['search']['log_dir']
        self.logger = logging.getLogger('Searching')
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if self.config['search']['save_log']:
            file_handler = logging.FileHandler(
                os.path.join(log_dir,'log_{}.txt'.format(time.strftime('%Y%m%d%H%M%S'))))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboardX_log'))
        return
        
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
        self.model = NasShell(in_channels=len(self.config['data']['all_mods']), 
                              init_n_kernels=self.config['search']['init_n_kernels'], 
                              out_channels=len(self.config['data']['labels']), 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              device=self.device,
                              normal_w_share=self.config['search']['normal_w_share'], 
                              channel_change=self.config['search']['channel_change']).to(self.device)
        
        self.logger.info('param size = %.3f MB', calc_param_size(self.model))
        
    def search(self):
        pass
    
    def train(self):
        pass
    
    def infer(self):
        pass
    
    def _log_clear(self):
        '''
        This is going to be put in the end of this class.
        '''
        while self.logger.handlers:
            self.logger.handlers.pop()
        return
    


def test():
    for i in tqdm(range(10)):
        print(i)
    
if __name__ == '__main__':
    search_network = Searching(jupyter = False)
#     test()
#     search_network.run()