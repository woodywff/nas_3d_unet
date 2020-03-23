import numpy as np
from torch.nn.functional import interpolate
import pdb
from graphviz import Digraph


def calc_param_size(model):
    '''
    Show the memory cost of model.parameters, in MB. 
    '''
    return np.sum(np.prod(v.size()) for v in model.parameters())*4e-6

def dim_assert(t_list):
    '''
    To make sure that all the tensors in t_list has the same dims.
    '''
    dims = tuple(np.max([t.size() for t in t_list], axis=0)[-3:])
    for i in range(len(t_list)):
        if tuple(t_list[i].shape[-3:]) != dims:
            print_red('inconsistent dim: i')
            t_list[i] = interpolate(t_list[i], dims)
    return t_list


def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))
    

def visualize(genotype, filename, caption=None, format='png'):
#     pdb.set_trace()
    g = Digraph(
        format=format,
        graph_attr = dict(dpi='800'),
        edge_attr = dict(fontsize='20'),
        node_attr = dict(style='filled', shape='rect', align='center',
                         fontsize='20', height='0.5', width='0.5',
                         penwidth='2'),
        engine='dot'
    )
    g.body.extend(['randkdir=LR'])

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i+1]:
            op, j = genotype[k]
            if j == 0:
                u = 'c_{k-2}'
            elif j == 1:
                u = 'c_{k-1}'
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')

    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), 'c_{k}', fillcolor='gray')
        
    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(filename, view=True)