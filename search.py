import pdb
import argparse
import yaml
from tqdm.notebook import tqdm

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
            from tqdm import tqdm
        
        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        for i in tqdm(range(10)):
            print(i)
        return
        
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
    from tqdm import tqdm
    search_network = Searching()
    test()
#     search_network.run()