
from Utils import *

Data_dir = "/home/luody/Perturb_seq/Data/Adamson"

# Read Data
## Matrix
mat=scipy.io.mmread(os.path.join(Data_dir, "GSM2406677_10X005_matrix.mtx.txt"))

genes_path = os.path.join(Data_dir, "GSM2406677_10X005_genes.tsv")
barcodes_path = os.path.join(Data_dir, "GSM2406681_10X010_barcodes.tsv")
gene_names = pd.read_table(genes_path, sep='\t', skiprows=0, header=False)
gene_names = gene_names.iloc[:,1]
barcodes = pd.read_table(barcodes_path, sep='\t', skiprows=0, header=False)
barcodes = list(barcodes.iloc[:,1])

Y=pd.DataFrame(mat.toarray())
Y.index=gene_names
Y.columns=barcodes
Y_log = np.log2(tp10k_transform(Y)+1)
## Guide (cell identity)
cell_ident = pd.read_csv(os.path.join(Data_dir, "GSM2406681_10X010_cell_identities.csv"))

