import pdb
import argparse
import yaml
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter
import logging
import os
import time

class Searching():
    '''
    Main class for searching
    '''
    def __init__(self, jupyter = True):
        self.jupyter = jupyter
        self._init_configure()
        self._init_logger()
        pass
    
    def _init_configure(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',type=str,default='config.yml',
                            help='Configuration file to use')
        if self.jupyter: # for jupyter notebook
            self.args = parser.parse_args(args=[])
        else:  # for shell
            self.args = parser.parse_args()
            from tqdm import tqdm
        
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
        
    def search(self):
        pass
    
    def train(self):
        pass
    
    def infer(self):
        pass
    
    def log_clear(self):
        while self.logger.handlers:
            self.logger.handlers.pop()
        return
    


def test():
    for i in tqdm(range(10)):
        print(i)
    
if __name__ == '__main__':
    from tqdm import tqdm
    search_network = Searching(jupyter = False)
    test()
#     search_network.run()