from collections import defaultdict
import logging

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm


class TadeClustering():
    # TODO long vector ?
    def main(self):
        self.transpose = False
        self.read_tade_mx()
        if self.transpose:
            self.mx = self.mx.T
        #self.dim_reduce()
        self.plot_clust(self.mx)

    def read_tade_mx(self):
        logging.info('Reading Tade matrix..')
        tade_d = defaultdict(int)
        with open('/mnt/store/hlt/Language/Hungarian/Dic/tade.tsv') as tade_f:
            for line in tade_f:
                verb, column, freq, _, _ = line.split()
                freq = int(freq)
                if freq:
                    for cas in column.split('_'):
                        tade_d[verb, cas.upper()] += freq
        self.verbs, self.cases = zip(*tade_d.keys())
        case_i = {cas: i for i, cas in enumerate(set(self.cases))}
        verb_i = {vrb: i for i, vrb in enumerate(set(self.verbs))}
        #self.mx =  lil_matrix((len(verb_i), len(case_i)))#, 'int')
        self.mx =  np.zeros((len(verb_i), len(case_i)), 'int')
        logging.info(self.mx.shape)
        for vrb, col in  tade_d.keys():
            self.mx[verb_i[vrb], case_i[col]] = tade_d[vrb, col] 
        #self.mx = self.mx[(self.mx != 0).sum(axis=0) > 20]
        #verb_sum = self.mx.sum(axis=1)
        #case_sum = self.mx.sum(axis=0)
        verb_argrank = (self.mx != 0).sum(axis=1).argsort()[::-1].reshape(-1,1) 
        case_argrank = (self.mx != 0).sum(axis=0).argsort()[::-1].reshape(1,-1)
        self.verbs = np.array(self.verbs)[verb_argrank]
        self.cases = np.array(self.cases)[case_argrank]
        logging.debug(type(self.mx))
        for a in self.mx, verb_argrank, case_argrank:
            logging.debug(a.shape)
        if self.transpose:
            self.mx = self.mx[case_argrank, verb_argrank]
        else:
            self.mx = self.mx[verb_argrank, case_argrank]
        logging.debug(self.mx.shape) 

    def dim_reduce(self):
        apply_tsne = False
        if self.mx.shape[1] > 500 or not apply_tsne:
            logging.info('PCA..')
            pca = PCA(n_components=200 if apply_tsne else 2)
            self.mx = pca.fit_transform(self.mx)
        logging.info('t-SNE..')
        if apply_tsne:
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
        logging.info('Argsorting mx cases..')
        self.mx = self.mx[:, np.argsort(clusser.column_labels_)]

    def plot_clust(self, mx):
        logging.info('Plotting..')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if issparse(self.mx): 
            pass
            #ax.spy(self.mx[:200,:200], marker='.')
            #ax.set_aspect('auto')
        elif True:
            cax = ax.matshow(self.mx[:10000,:20], norm=LogNorm())
            fig.colorbar(cax)
            ax.set_aspect('auto')
        else:
            ax.scatter(*zip(*mx))
            for label, row in zip(self.verbs if self.transpose else self.cases, self.mx):
                ax.annotate(label, xy = row)
        if False:
            plt.show()
        else:
            plt.savefig('tade.pdf')


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    TadeClustering().main()
