from collections import defaultdict
import logging
from os.path import isfile

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm


class TadeClustering():
    # TODO long vector ?
    def main(self, transpose=False):
        self.mx_filen = 'tade_mx.npz'
        self.read_tade_mx()
        if transpose:
            logging.info('Transposing')
            self.mx = self.mx.T
        self.mut_info()
        self.sort_lines()#cut_off=False)
        #self.cocluster()
        self.dim_reduce()#apply_tsne=False)
        np.savez(self.mx_filen, self.mx, self.cases, self.verbs)
        self.plot_tade(self.mx, transpose)#, savefig=True) 

    def read_tade_mx(self, load=False):
        if load: 
            logging.debug(self.mx_filen)
            if isfile(self.mx_filen):
                logging.info('Loading mx..')
                self.mx, self.cases, self.verbs = np.load(self.mx_filen)
                logging.debug(type(self.mx))
                return
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
        self.mx =  np.zeros((len(verb_i), len(case_i)))
        logging.info(self.mx.shape)
        for vrb, col in  tade_d.keys():
            self.mx[verb_i[vrb], case_i[col]] += tade_d[vrb, col] 

    def mut_info(self):
        logging.info('Computing mutual information..')
        self.mx += 1
        n_token =  self.mx.sum()
        verb_freq = self.mx.sum(axis=1).reshape(-1,1)
        case_freq = self.mx.sum(axis=0).reshape(1,-1)
        self.mx /= verb_freq
        self.mx /= case_freq
        self.mx *= n_token
        self.mx = np.log(self.mx)

    def sort_lines(self, cut_off=True):
        verb_argrank = (self.mx != 0).sum(axis=1).argsort()[::-1]
        case_argrank = (self.mx != 0).sum(axis=0).argsort()[::-1]
        if cut_off:
            verb_argrank = verb_argrank[:10000]
            case_argrank = case_argrank[:20]
        verb_argrank = verb_argrank.reshape(-1,1)
        case_argrank = case_argrank.reshape(1,-1) 
        self.verbs = np.array(self.verbs)[verb_argrank]
        self.cases = np.array(self.cases)[case_argrank]
        self.mx = self.mx[verb_argrank, case_argrank]

    def cocluster(self, blockdiag=True):
        logging.info('Co-clustering Tade..')
        if blockdiag:
            logging.info('blockdiag')
            clusser = SpectralCoclustering(n_jobs=-1)
        else: # checkerboard
            logging.info('checkerboard')
            clusser = SpectralBiclustering(n_jobs=-1)
            #n_clusters=3, svd_method='randomized',
        clusser.fit(self.mx)
        logging.info('Argsorting mx rows..')
        self.mx = self.mx[np.argsort(clusser.row_labels_)]
        logging.info('Argsorting mx cases..')
        self.mx = self.mx[:, np.argsort(clusser.column_labels_)]

    def dim_reduce(self, apply_tsne=True):
        if self.mx.shape[1] > 400 or not apply_tsne:
            logging.info('PCA.. from {}'.format(self.mx.shape))
            pca = PCA(n_components=200 if apply_tsne else 2)
            self.mx = pca.fit_transform(self.mx)
        if apply_tsne:
            logging.info('t-SNE.. from {}'.format(self.mx.shape))
            tsne = TSNE(init='pca')
            self.mx = tsne.fit_transform(self.mx)
        # method : string (default: 'barnes_hut') 
        #   By default the gradient calculation algorithm uses Barnes-Hut
        #   approximation running in O(NlogN) time. method='exact' will run on
        #   the slower, but exact, algorithm in O(N^2) time

    def plot_tade(self, mx, transpose, savefig=False):
        logging.info('Plotting..')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if issparse(self.mx): 
            ax.spy(self.mx[:200,:200], marker='.')
            ax.set_aspect('auto')
        elif self.mx.shape[1] > 2:
            cax = ax.matshow(self.mx)#, norm=LogNorm())
            fig.colorbar(cax)
            ax.set_aspect('auto')
        else:
            ax.scatter(*zip(*mx))
            lab_col = self.cases if transpose else self.verbs
            for label1, row in zip(lab_col.reshape(-1), self.mx[:100]):
                ax.annotate(label1, xy=row, fontsize=10)
        if savefig:
            logging.info('Saving fig..')
            plt.savefig('tade-mutinfo-tsne.pdf')
        else:
            plt.show() 


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    TadeClustering().main()
