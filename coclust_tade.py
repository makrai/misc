from collections import defaultdict
import logging

from matplotlib import pyplot as plt 
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, issparse
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TadeClustering():
    # TODO long vector, PCA to 200, t-SNE to 2
    def main(self):
        self.read_tade_mx()
        self.dim_reduce()
        self.plot_clust(self.mx)

    def read_tade_mx(self):
        logging.info('Reading Tade matrix..')
        tade_d = defaultdict(int)
        with open('/mnt/store/hlt/Language/Hungarian/Dic/tade.tsv') as tade_f:
            for line in tade_f:
                verb, column, freq, _, _ = line.split()
                freq = int(freq)
                if freq > 1:
                    for cas in column.split('_'):
                        tade_d[verb, cas] += freq
        verbs, columns = zip(*tade_d.keys())
        column_i = {col: i for i, col in enumerate(set(columns))}
        verb_i = {vrb: i for i, vrb in enumerate(set(verbs))}
        #self.mx =  lil_matrix((len(verb_i), len(column_i)))#, 'int')
        self.mx =  np.zeros((len(verb_i), len(column_i)), 'int')
        logging.info(self.mx.shape)
        for vrb, col in  tade_d.keys():
            self.mx[verb_i[vrb], column_i[col]] = tade_d[vrb, col] 

    def dim_reduce(self):
        logging.info('PCA..')
        pca = PCA(n_components=200)
        self.mx = pca.fit_transform(self.mx)
        logging.info('t-SNE..')
        tsne = TSNE(init='pca')
        self.mx = tsne.fit_transform(self.mx)
        # method : string (default: 'barnes_hut') 
        #   By default the gradient calculation algorithm uses Barnes-Hut
        #   approximation running in O(NlogN) time. method='exact' will run on
        #   the slower, but exact, algorithm in O(N^2) time

    def cocluster(self):
        logging.info('Co-clustering Tade..')
        clusser = SpectralCoclustering(n_jobs=-1)
        #clusser = SpectralBiclustering(n_jobs=-1)
        #n_clusters=3, svd_method='randomized',
        clusser.fit(self.mx)
        logging.info('Argsorting mx rows..')
        self.mx = self.mx[np.argsort(clusser.row_labels_)]
        logging.info('Argsorting mx columns..')
        self.mx = self.mx[:, np.argsort(clusser.column_labels_)]

    def plot_clust(self, mx):
        logging.info('Plotting..')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.spy(mx, marker='.')
        ax.plot(mx)
        ax.set_aspect('auto')
        plt.show()


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    TadeClustering().main()
