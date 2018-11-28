from Utils import *
'''
On Adamson data
'''
Data_dir = "/home/luodongyang/SCData/Perturb/Adamson/"
#------------------------------------------------------------------------#
# Read Data
## Matrix
mat=mmread(os.path.join(Data_dir, "GSM2406677_10X005_matrix.mtx.txt"))
cell_ident = pd.read_csv(os.path.join(Data_dir, "GSM2406677_10X005_cell_identities.csv"))
genes_path = os.path.join(Data_dir, "GSM2406677_10X005_genes.tsv")
barcodes_path = os.path.join(Data_dir, "GSM2406677_10X005_barcodes.tsv")
gene_names = pd.read_table(genes_path, sep='\t', skiprows=0, header=None)
gene_names = gene_names.iloc[:,1]
barcodes = pd.read_table(barcodes_path, sep='\t', skiprows=0, header=None)
barcodes = list(barcodes.iloc[:,0])
#------------------------------------------------------------------------#
# Processing
## conversion & Filtering
guide_summ = guide_summary(cell_ident) # Guide summary
selected_guides = list(guide_summ['GuideName'][guide_summ['Count'] > 100])
temp_idx = []
for ll in range(len(cell_ident)):
    if cell_ident['guide identity'][ll] in selected_guides:
        temp_idx.append(ll)
cell_ident = cell_ident.loc[temp_idx]
Y = pd.DataFrame(mat.toarray())
Y.index = gene_names
Y.columns = barcodes
[filtered_genes,filtered_cells] = filter_Gene_Cell(Y, gene_thresh=10, cell_thresh=1000) # filtering
selected_cells = list(set(filtered_cells) & set(cell_ident['cell BC']))
cell_ident.index = cell_ident['cell BC']
cell_ident = cell_ident.loc[selected_cells]
Y = Y.loc[filtered_genes, selected_cells]
Y_log = pd.DataFrame(np.log2(tp10k_transform(Y)+1))
guides = cell_ident['guide identity']
#------------------------------------------------------------------------#
# PCA
[Ufb,Sfb,Vfb,PCscore] = fb_pca(Y_log, n_components=50, center=True, scale=False)
## PC variance explained
plt.plot(Sfb, label='PC Variance Explained')
plt.savefig('./Figs/PC_eigens.jpg', dpi=300)
plt.close()

## Use PC scores for plotting
plot_pca = PCscore[['PC1','PC2']]
plot_pca['Guides'] = guides
sns.lmplot('PC1','PC2',data=plot_pca,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/PCA.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# t-SNE
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne = tsne_model.fit_transform(PCscore.iloc[:,range(10)])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# LASSO 
X = pd.DataFrame(Vfb.transpose())
X.index = PCscore.index
X.columns = PCscore.columns
guides_dummy = pd.get_dummies(guides)
lasso_model = linear_model.Lasso(alpha=0.1, precompute=True)
lasso_model.fit(PCscore, guides_dummy)

#------------------------------------------------------------------------#
# Random Forest
guides_dummy = pd.get_dummies(guides)
RF_model = RandomForestClassifier(n_estimators=100,n_jobs=-1,oob_score=True,class_weight='balanced')
RF_model.fit(PCscore, guides_dummy)

PC_rank = pd.DataFrame({'PCs':['PC'+str(x+1) for x in range(50)],
                        'Importance':RF_model.feature_importances_})
PC_rank = PC_rank.loc[np.argsort(-PC_rank['Importance'], )]
PC_rank.index = range(1,51)
plt.plot(PC_rank['Importance'], label='PC Importance')
plt.savefig('./Figs/PC_importance.jpg', dpi=300)
plt.close()
PC_rank.to_csv('./Figs/PC_importance.csv')
#------------------------------------------------------------------------#
# PCA with important PCs
selected_PCs = list(PC_rank['PCs'][0:30]) # Previous = 10
New_feature_Y = PCscore[selected_PCs].transpose()
[Unew,Snew,Vnew,PCscore_new] = fb_pca(New_feature_Y, n_components=10, center=True, scale=False)
plot_pca = PCscore_new[['PC1','PC2']]
plot_pca['Guides'] = guides
sns.lmplot('PC1','PC2',data=plot_pca,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/PCA_new.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# tSNE with important PCs
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne = tsne_model.fit_transform(PCscore[selected_PCs])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE_new.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# tSNE with important PCs
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
selected_PCs = list(set(selected_PCs) - set(['PC'+str(x) for x in range(1,5)]))
T_sne = tsne_model.fit_transform(PCscore[selected_PCs])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE_PC1-4_removed.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
'''
On Dixit data
'''
from Utils import *
Data_dir = "/home/luodongyang/SCData/Perturb/Dixit/"
#------------------------------------------------------------------------#
# Read Data
## Matrix
mat=mmread(os.path.join(Data_dir, "GSM2396856_dc_3hr.mtx.txt"))
genes_path = os.path.join(Data_dir, "GSM2396856_dc_3hr_genenames.csv")
gene_names = pd.read_table(genes_path, sep=',', skiprows=0)
gene_names = gene_names.iloc[:,1]
barcodes_path = os.path.join(Data_dir, "GSM2396856_dc_3hr_cellnames.csv")
barcodes = pd.read_table(barcodes_path, sep=',', skiprows=0)
barcodes = list(barcodes.iloc[:,1])
## Get the GUIDE part of the X
cbc_gbc_dict_path = os.path.join(Data_dir, "GSM2396856_dc_3hr_cbc_gbc_dict_lenient.csv")
gbcs = [row[0] for row in csv.reader(open(cbc_gbc_dict_path))]
cbcs_raw = [row[1] for row in csv.reader(open(cbc_gbc_dict_path))]
cbcs = []
for temp_val in cbcs_raw:
    temp = temp_val.replace(' ','').split(',')
    cbcs.append(list(set(temp)&set(barcodes)))
gbc_cbc_dict = dict(zip(gbcs, cbcs))
X_guides = dict2X(GUIDES_DICT=gbc_cbc_dict, cbcs=barcodes)
#------------------------------------------------------------------------#
# Processing
## conversion & Filtering
Y = pd.DataFrame(mat.toarray())
Y.index = gene_names
Y.columns = barcodes
[filtered_genes,filtered_cells] = filter_Gene_Cell(Y, gene_thresh=10, cell_thresh=1000) # filtering
cell_idx = X_guides.index[X_guides.sum(axis=1)==1]
selected_cells = list(set(filtered_cells) & set(cell_idx))
Y = Y.loc[filtered_genes, selected_cells]
X_guides = X_guides.loc[selected_cells]
Y_log = pd.DataFrame(np.log2(tp10k_transform(Y)+1))

guide_list = list(X_guides.columns)
guides = []
for ii in range(len(X_guides)):
    guides.append(guide_list[list(X_guides.iloc[ii,:]).index(1)])
#------------------------------------------------------------------------#
# Merge Guides --> same gene
for ii in range(len(guides)):
    guides[ii] = guides[ii].split('_')[1]
#------------------------------------------------------------------------#
# PCA
[Ufb,Sfb,Vfb,PCscore] = fb_pca(Y_log, n_components=100, center=True, scale=False)
## PC variance explained
plt.plot(Sfb, label='PC Variance Explained')
plt.savefig('./Figs/PC_eigens_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
## Use PC scores for plotting
plot_pca = PCscore[['PC1','PC2']]
plot_pca['Guides'] = guides
sns.lmplot('PC1','PC2',data=plot_pca,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/PCA_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# t-SNE
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne = tsne_model.fit_transform(PCscore.iloc[:,range(15)])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# LASSO 
'''
X = pd.DataFrame(Vfb.transpose())
X.index = PCscore.index
X.columns = PCscore.columns
guides_dummy = pd.get_dummies(guides)
lasso_model = linear_model.Lasso(alpha=0.1, precompute=True)
lasso_model.fit(PCscore, guides_dummy)
'''
#------------------------------------------------------------------------#
# Random Forest
guides_dummy = pd.get_dummies(guides)
RF_model = RandomForestClassifier(n_estimators=100,n_jobs=-1,oob_score=True,class_weight='balanced')
RF_model.fit(PCscore, guides_dummy)

PC_rank = pd.DataFrame({'PCs':['PC'+str(x+1) for x in range(100)],
                        'Importance':RF_model.feature_importances_})
PC_rank = PC_rank.loc[np.argsort(-PC_rank['Importance'], )]
PC_rank.index = range(1,101)
plt.plot(PC_rank['Importance'], label='PC Importance')
plt.savefig('./Figs/PC_importance_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
PC_rank.to_csv('./Figs/PC_importance_Dixit_merge_xxxGuide.csv')
#------------------------------------------------------------------------#
# PCA with important PCs
selected_PCs = list(PC_rank['PCs'][0:10])
New_feature_Y = PCscore[selected_PCs].transpose()
[Unew,Snew,Vnew,PCscore_new] = fb_pca(New_feature_Y, n_components=10, center=True, scale=False)
plot_pca = PCscore_new[['PC1','PC2']]
plot_pca['Guides'] = guides
sns.lmplot('PC1','PC2',data=plot_pca,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/PCA_new_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# tSNE with important PCs
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne = tsne_model.fit_transform(PCscore[selected_PCs])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE_new_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# tSNE with important PCs
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
selected_PCs = list(set(selected_PCs) - set(['PC'+str(x) for x in range(1,10)]))
T_sne = tsne_model.fit_transform(PCscore[selected_PCs])
T_sne = pd.DataFrame(T_sne)
plot_tsne = T_sne.copy()
plot_tsne.columns = ['tSNE-1', 'tSNE-2']
plot_tsne.index = selected_cells
plot_tsne['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_tsne,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./Figs/tSNE_PC1-4_removed_Dixit_merge_xxxGuide.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#

