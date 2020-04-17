import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from graphviz import Digraph
import matplotlib.image as mi
import glob
import pdb
    
def plot_cell(filename, n_nodes=3, dc=True, fmt='png', dpi='200'):
    '''
    Draw the architecture of downward or upward cell for searching.
    filename: save path
    n_nodes: number of nodes
    dc: if True, for downward cell, otherwise for upward cell
    '''
    g = Digraph(format=fmt,
                graph_attr = dict(dpi=dpi),
                edge_attr = dict(fontsize='20',penwidth='1.5'),
                node_attr = dict(style='filled', shape='rect', align='center',
                                 fontsize='20', height='0.1', width='0.1',
                                 penwidth='2'))
                #engine='dot')
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
    return

def plot_searched_cell(op_list, filename, dc=True, fmt='png', dpi='200'):
    '''
    Draw searched downward and upward cells.
    filename: save path
    op_list: list of tuple (op_name: str, input_index: int)
    dc: if True, for downward cell, otherwise for upward cell
    '''
    g = Digraph(format=fmt,
                graph_attr = dict(dpi=dpi),
                edge_attr = dict(fontsize='20',penwidth='1.5'),
                node_attr = dict(style='filled', shape='rect', align='center',
                                 fontsize='20', height='0.1', width='0.1',
                                 penwidth='2'))
                #engine='dot')
    g.attr(rankdir='TB' if dc else 'BT')
    g.node('x0', label='X0', fillcolor='white',shape='plaintext')
    g.node('x1', label='X1', fillcolor='white', shape='plaintext')
    g.node('p0', label='pre0(s=2)' if dc else 'pre0(s=1)', fillcolor='ghostwhite')
    g.node('p1', label='pre1(s=1)', fillcolor='ghostwhite')
    g.edge('x0', 'p0')
    g.edge('x1', 'p1')
    assert len(op_list) % 2 == 0
    n_nodes = len(op_list) // 2
    
    xs = ['p0','p1']
    i = 0
    for node_i in range(n_nodes):
        with g.subgraph(name='cluster_{}'.format(node_i)) as sg:
            sg.attr(style='dashed', color='red', label='node {}'.format(node_i), 
                    fontsize='20', penwidth='1.8', fontcolor='red')
            name_add = 'add'+str(node_i)
            sg.node(name=name_add, label='+', fillcolor='lightskyblue2')
            with sg.subgraph() as ssg:
                ssg.attr(rank='same')
                for _ in range(2):
                    op, x_i = op_list[i]
                    ssg.node(str(i), label=op, fillcolor='ghostwhite')
                    g.edge(xs[x_i], str(i))
                    g.edge(str(i),name_add)
                    i += 1
        xs.append(name_add)
        
    g.node('concat', label='C', fillcolor='wheat')
    for name in xs[-3:]:
        g.edge(name, 'concat')
    g.node('y', label='Y', fillcolor='white', shape='plaintext')
    g.edge('concat','y')
    g.render(filename)
    return

from prim_ops import UpOps

def plot_ops(filename, fmt='png', dpi='200'):
    '''
    Draw the alpha update mechanism figure.
    '''
    g = Digraph(format=fmt,
                graph_attr = dict(dpi=dpi),
                edge_attr = dict(fontsize='20',penwidth='1.5'),
                node_attr = dict(style='filled', shape='rect', align='center',
                                 fontsize='20', height='0.1', width='0.1',
                                 penwidth='2'))
    g.node('x', label='X', fillcolor='white',shape='plaintext')
    g.node('add', label='+', fillcolor='lightskyblue2')
    with g.subgraph() as sg:
        sg.attr(rank='same')
        for op in UpOps:
            sg.node(op, fillcolor='ghostwhite')
            g.edge('x',op)
            g.edge(op,'add')
    g.node('y', label='Y', fillcolor='white', shape='plaintext')
    g.edge('add','y')
    g.render(filename)
    return    
    

def evaluation_plot(csv_file, criteria, label, save_name, val=True):
    df = pd.read_csv(csv_file)
    dict_criteria = {}
    dict_criteria['ET'] = [x for x in df[criteria+'_ET'] if not np.isnan(x)][:-5]
    dict_criteria['TC'] = [x for x in df[criteria+'_TC'] if not np.isnan(x)][:-5]
    dict_criteria['WT'] = [x for x in df[criteria+'_WT'] if not np.isnan(x)][:-5]
    plt.figure()
    plt.boxplot(dict_criteria.values(),
                labels=[key+'\nmean: %.2f'%(np.mean(dict_criteria[key])) for key in dict_criteria.keys()])
    plt.ylabel(label)
    dataset_type = 'Val' if val else 'Training'
    plt.title(label + ' Boxplot of ' + dataset_type + ' Dataset')
    plt.savefig(save_name,dpi=200)
    
def draw_evaluate(csv_file, save_dir,val=True, fig_format='png'):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    evaluation_plot(csv_file, 'Dice', 'Dice Coefficient', os.path.join(save_dir,'dice_val.'+fig_format), val=val)
    evaluation_plot(csv_file, 'Sensitivity', 'Sensitivity', os.path.join(save_dir,'sensitivity_val.'+fig_format), val=val)
    evaluation_plot(csv_file, 'Specificity', 'Specificity', os.path.join(save_dir,'specificity_val.'+fig_format), val=val)
    evaluation_plot(csv_file, 'Hausdorff95', 'Hausdorff Disdance', os.path.join(save_dir,'hausdorff_val.'+fig_format), val=val)
    
def four_in_all(png_fold, fig_format='pdf'):
    files = glob.glob(os.path.join(png_fold,'*'))
    plt.figure()
    fig, axs = plt.subplots(2, 2,figsize=(15,15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.0, hspace=0.0)
    i = 0
    for row in range(2):
        for col in range(2):
            img = mi.imread(files[i])
            axs[row,col].imshow(img)
            axs[row,col].axis('off')
            i += 1
    fig.savefig(os.path.join(png_fold,'four_in_all.'+fig_format),dpi=200)
    return

if __name__ == '__main__':
    draw_evaluate('data/results/Stats_Training_final.csv','log/training_figs', val=False)
    draw_evaluate('data/results/Stats_Validation_final.csv','log/val_figs')
    four_in_all('log/training_figs')
    four_in_all('log/val_figs')
    
    import pickle
    from genotype import Genotype

    plot_cell('log/dc')
    plot_cell('log/uc', dc=False)

    with open('BACKUP/new_best_genotype.pkl','rb') as f:
        g = eval(pickle.load(f)[0])
    plot_searched_cell(g.down, 'log/searched_dc')
    plot_searched_cell(g.up, 'log/searched_uc', dc=False)

    plot_ops('log/ops')