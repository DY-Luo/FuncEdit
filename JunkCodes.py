
### Use Vfb scores for plotting
temp = (pd.DataFrame(Vfb)).transpose()
temp.index = Y_log.columns
temp.columns = ['PC'+str(x+1) for x in range(50)]
plt.scatter(temp['PC1'], temp['PC2'], s=5, alpha=0.5)
plt.savefig('./PCA2.jpg', dpi=300)
plt.close()

import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import *

rand_data = np.random.normal(size=(100,2))
plt.scatter(rand_data[:,0], rand_data[:,1])
plt.savefig('./test2.jpg')




Vfb_cluster=Vfb.ix[:,range(9)]
model = sklearn.manifold.TSNE(n_components=2, perplexity=20,verbose=2,init='pca',n_iter_without_progress=10000,min_grad_norm=0)
T_sne=model.fit_transform(Vfb_cluster)
T_sne=pd.DataFrame(T_sne)
plt.scatter(T_sne[0],T_sne[1],alpha=0.75,c=cqc,cmap='bwr')
Tsneplot=T_sne.copy()
Tsneplot.columns=['1','2']
Tsneplot['clustering']=np.array(info_labels)
sns.lmplot('1','2',data=Tsneplot,hue='clustering',fit_reg=False)
