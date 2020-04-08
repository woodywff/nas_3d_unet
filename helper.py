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
    
    
def plot_cell(filename, n_nodes=3, dc=True, fmt='png'):
    '''
    filename: save path
    n_nodes: number of nodes
    dc: if True, for downward cell, otherwise for upward cell
    '''
    g = Digraph(format=fmt,
                graph_attr = dict(dpi='800'),
                edge_attr = dict(fontsize='20',penwidth='1.5'),
                node_attr = dict(style='filled', shape='rect', align='center',
                                 fontsize='20', height='0.1', width='0.1',
                                 penwidth='2'),
                engine='dot')
    g.attr(rankdir='TB' if dc else 'BT')
    g.node('x0', label='X0', fillcolor='white',shape='plaintext')
    g.node('x1', label='X1', fillcolor='white', shape='plaintext')
    g.node('p0', label='pre0(s=2)' if dc else 'pre0(s=1)', fillcolor='ghostwhite')
    g.node('p1', label='pre1(s=1)', fillcolor='ghostwhite')
    g.edge('x0', 'p0')
    g.edge('x1', 'p1')
    
    xs = ['p0','p1']
    for n_ops in range(2,2+n_nodes):
        node_i = n_ops-2
        with g.subgraph(name='cluster_{}'.format(node_i)) as sg:
            sg.attr(style='dashed', color='red', label='node {}'.format(node_i), 
                    fontsize='20', penwidth='1.8', fontcolor='red')
            name_add = 'add'+str(node_i)
            sg.node(name=name_add, label='+', fillcolor='lightskyblue2')
            with sg.subgraph() as ssg:
                ssg.attr(rank='same')
                for op_i in range(n_ops):
                    name_op = 'n{}_{}'.format(node_i, op_i)
                    if dc and op_i < 2:
                        label = 'D'
                    elif not dc and op_i == 1:
                        label = 'U'
                    else:
                        label = 'N'
                    ssg.node(name=name_op, label=label, fillcolor='ghostwhite')
                    g.edge(xs[op_i], name_op)
                    g.edge(name_op, name_add)
        xs.append(name_add)
        
    g.node('concat', label='C', fillcolor='wheat')
    for name in xs[-3:]:
        g.edge(name, 'concat')
    g.node('y', label='Y', fillcolor='white', shape='plaintext')
    g.edge('concat','y')
    g.render(filename)

def plot_searched_cell(genotype, filename, dc=True, fmt='png'):
    '''
    Draw searched downward and upward cells.
    filename: save path
    genotype: Genotype.down or Genotype.up, Genotype is defined in genotype.py
    dc: if True, for downward cell, otherwise for upward cell
    '''
#     pdb.set_trace()
    g = Digraph(format=fmt,
                graph_attr = dict(dpi='800'),
                edge_attr = dict(fontsize='20'),
                node_attr = dict(style='filled', shape='rect', align='center',
                                 fontsize='20', height='0.5', width='0.5',
                                 penwidth='2'),
                engine='dot')
    g.attr(rankdir='TB' if dc else 'BT')
    g.node('x0', fillcolor='darkseagreen2',shape='plaintext')
    g.node('x1', fillcolor='darkseagreen2',shape='plaintext')
    g.node('p0', label='pre0', fillcolor='ghostwhite')
    g.node('p1', label='pre1', fillcolor='ghostwhite')
    g.edge('x0', 'p0')
    g.edge('x1', 'p1')
    assert len(genotype) % 2 == 0
    n_nodes = len(genotype) // 2
    
    xs = ['p0','p1']
    i = 0
    for node in range(n_nodes):
        with g.subgraph(name='cluster_{}'.format(node)) as sg:
            sg.attr(style='dashed', color='red', label='node {}'.format(node))
            sg.node('add'+str(node), fillcolor='lightblue')
            with sg.subgraph() as ssg:
                ssg.attr(rank='same')
                for _ in range(2):
                    op, x_i = genotype[i]
                    ssg.node(str(i), label=op, fillcolor='ghostwhite')
                    g.edge(xs[x_i], str(i))
                    g.edge(str(i),'add'+str(node))
                    i += 1
        xs.append('add'+str(node))
        
    g.node('concat', label='C', fillcolor='palegoldenrod')
    for name in xs[-3:]:
        g.edge(name, 'concat')
    g.node('y', fillcolor='cyan3', shape='plaintext')
    g.edge('concat','y')
    
    g.attr(label='DC' if dc else 'UC', overlap='false', fontsize='20', fontname='times')

    g.render(filename)
    