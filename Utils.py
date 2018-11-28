# Utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.io import mmread
import numpy as np
import pandas as pd
import seaborn as sns
import fbpca
import csv
from sklearn import linear_model
from sklearn import decomposition
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier

def tp10k_transform(GEMat,norm_factor=1.0e4):
    """normalize columns of pandas dataframe to sum to a constant, by default 10,000"""
    return(norm_factor*(GEMat / GEMat.sum()))

def filter_Gene_Cell(GEMat, gene_thresh=10, cell_thresh=1000):
    """col: genes, row: cells"""
    """gene with > [gene_thresh] cells"""
    """cells with > [cell_thresh] UMIs"""
    #temp_GEMat = GEMat.copy()
    gene_list = GEMat.index
    cell_list = GEMat.columns
    selected_genes = gene_list[(GEMat > 0).sum(axis=1) > gene_thresh]
    selected_cells = cell_list[GEMat.sum(axis=0) > cell_thresh]
    #temp_GEMat = temp_GEMat.loc[selected_genes, selected_cells]
    return selected_genes, selected_cells

def guide_summary(ident_df, col_name='guide identity'):
    ## Summary the number of different guides
    guide_list = list(np.unique(ident_df[col_name]))
    guide_count = []
    for nn in guide_list:
        guide_count.append(sum(ident_df[col_name]==nn))
    return pd.DataFrame({"GuideName":guide_list, "Count":guide_count})

def centralization(GEMat):
    return np.matrix(GEMat.subtract(GEMat.mean(axis=1),axis=0))

def scale(GEMat):
    return np.matrix(GEMat.divide(GEMat.std(axis=1),axis=0))

def fb_pca(GEMat, n_components, center=True, scale=False):
    col_names = GEMat.columns
    if center==True:
        GEMat = centralization(GEMat)
    if scale==True:
        GEMat = scale(GEMat)
    [Ufb,Sfb,Vfb] = fbpca.pca(GEMat, n_components)
    PCscore = pd.DataFrame(np.multiply(Vfb, Sfb.reshape(n_components,1)))
    PCscore.index = ['PC'+str(x+1) for x in range(n_components)]
    PCscore.columns = col_names
    return Ufb, Sfb, Vfb, PCscore.transpose()

def dict2X(GUIDES_DICT,cbcs):
    """convert guide cbc dictionary into covariate matrix"""
    X=pd.DataFrame()
    for key in GUIDES_DICT.keys():
        curkey=[]
        for cbc in cbcs:
            if cbc in GUIDES_DICT[key]:
                curkey.append(1)
            else:
                curkey.append(0)
        X[key]=np.array(curkey)
    X.index=cbcs
    return X


