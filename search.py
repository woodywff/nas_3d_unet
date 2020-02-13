import pdb
import argparse
import yaml

class Searching():
    '''
    Main class for searching
    '''
    def __init__(self, jupyter = False):
        self.jupyter = jupyter
        self._init_configure()
        pass
    
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
            print(self.config)
    
    def search(self):
        pass
    
    def train(self):
        pass
    
    def infer(self):
        pass
    
if __name__ == '__main__':
    search_network = Searching()
#     search_network.run()