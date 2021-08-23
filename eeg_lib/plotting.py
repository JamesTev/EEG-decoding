import numpy as np
import matplotlib.pyplot as plt

def get_x_offsets(n, width=0.2):
    "get x offsets of bar centres for grouped bar charts in matplotlib"
    if n%2 == 0: # even
        sides = [width*x/2 for x in range(1, n+1, 2)]
        return [-1*x for x in sides[::-1]] + sides
    else: # odd
        sides = [width*x/2 for x in range(3, n+1, 2)]
        return [-1*x for x in sides[::-1]] + [0] + sides

    
def grouped_bar(x, Y, width=0.2, xlabel='', ylabel='', legend=None, ax=None):
    """
    Expects Y to be num_samples x num_vars.
    
    x should be a vector repr. the independent variable
    """
    fig = None
    if ax is None:    
        fig, ax = plt.subplots(1, figsize=(16, 6))
    
    x_offsets = get_x_offsets(Y.shape[1], width=width*0.75)
    
    for i in range(len(x_offsets)):
        ax.bar(np.arange(len(x))+x_offsets[i], Y[:, i], width=width)
    
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.6)
    
    if legend is not None:
        ax.legend(legend)
                
    return fig, ax