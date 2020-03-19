from collections import namedtuple
import numpy as np
from prim_ops import DownOps, UpOps, NormOps
import pdb

Genotype = namedtuple('Genotype', 'down down_concat up up_concat'.split())

class GenoParser:
    def __init__(self, n_nodes):
        '''
        This is the class for genotype operations.
        n_nodes: How many nodes in a cell.
        '''
        self.n_nodes = n_nodes
        
    def parse(self, alpha1, alpha2, downward=True):
        '''
        alpha1: Weights for MixedOps with stride=1
        alpha2: Weights for MixedOps with stride=2
        Note these two matrix are the same as in nas.KernelNet().
        
        '''
        i = 0
        res = []
        for n_edges in range(2, 2 + self.n_nodes):
            gene = []
            for edge in range(n_edges):
                if downward and edge < 2:
                    argmax = np.argmax(alpha2[i])
                    gene.append((alpha2[i][argmax]*len(DownOps)/len(NormOps), DownOps[argmax], edge))
                elif not downward and edge == 1:
                    argmax = np.argmax(alpha2[i])
                    gene.append((alpha2[i][argmax]*len(UpOps)/len(NormOps), UpOps[argmax], edge))
                else:
                    argmax = np.argmax(alpha1[i])
                    gene.append((alpha1[i][argmax], NormOps[argmax], edge))
                i += 1
            gene.sort()
            res += [(op[1], op[2]) for op in gene[-2:]]
        return res