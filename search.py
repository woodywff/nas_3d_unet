import pdb
import argparse
import yaml

class SearchingProcess():
    '''
    Main class for searching
    '''
    def __init__(self):
        self._init_configure()
        pass
    
    def _init_configure(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',type=str,default='config.yml',
                            help='Configuration file to use')
        # for notebook:
        self.args = parser.parse_args(args=[])
        # for shell:
        #self.args = parser.parse_args()
        
        with open(self.args.config) as f:
            self.config = yaml.load(f)
    
    def search(self):
        pass
    
    def train(self):
        pass
    
    def infer(self):
        pass