
from Utils import *

Data_dir = "/home/luodongyang/SCData/Perturb/Adamson/"
#------------------------------------------------------------------------#
# Read Data
'''
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
Y = pd.DataFrame(mat.toarray())
Y.index = gene_names
Y.columns = barcodes
[filtered_genes,filtered_cells] = filter_Gene_Cell(Y, gene_thresh=10, cell_thresh=1000) # filtering
selected_cells = list(set(filtered_cells) & set(cell_ident['cell BC']))
cell_ident.index = cell_ident['cell BC']
cell_ident = cell_ident.loc[selected_cells]
Y = Y.loc[filtered_genes, selected_cells]
Y_log = pd.DataFrame(np.log2(tp10k_transform(Y)+1))
guide_summ = guide_summary(cell_ident) # Guide summary
guides = cell_ident['guide identity']
'''
## Small data for testing codes
Y = pd.DataFrame(np.concatenate((np.random.normal(0,1,(80,100)), np.random.uniform(0,5,(80,100))), axis=1))
Y = Y - np.min(Y)
Y.columns = ['Cell-'+str(x) for x in range(200)]
Y.index = ['Gene-'+str(x) for x in range(80)]
guides = ['guide-B']*10 + ['guide-A']*90 + ['guide-A']*5 + ['guide-B']*95
Y_log = pd.DataFrame(np.log2(tp10k_transform(Y)+1))

#------------------------------------------------------------------------#
# PCA
[Ufb,Sfb,Vfb,PCscore] = fb_pca(Y_log, n_components=50, center=True, scale=False)
## Use PC scores for plotting
plot_pca = PCscore[['PC1','PC2']]
plot_pca['Guides'] = guides
sns.lmplot('PC1','PC2',data=plot_pca,hue='Guides',fit_reg=False, scatter_kws={'s':5})
plt.savefig('./PCA_test.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# t-SNE
tsne_model = manifold.TSNE(n_components=2, perplexity=20, verbose=2,
                        init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne = tsne_model.fit_transform(PCscore.iloc[:,range(9)])
T_sne = pd.DataFrame(T_sne)
plot_data = T_sne.copy()
plot_data.columns = ['tSNE-1', 'tSNE-2']
plot_data['Guides'] = guides
sns.lmplot('tSNE-1','tSNE-2',data=plot_data,hue='Guides',fit_reg=False, size=10)
plt.savefig('./tSNE_test.jpg', dpi=300)
plt.close()
#------------------------------------------------------------------------#
# LASSO 
guides_dummy = pd.get_dummies(guides)
lasso_model = linear_model.Lasso(alpha=0.1, precompute=True)
lasso_model.fit(PCscore, guides_dummy)
print(lasso_model.coef_)
#------------------------------------------------------------------------#
# Random Forest

#------------------------------------------------------------------------#
