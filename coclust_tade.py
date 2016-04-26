from collections import defaultdict
import logging

from matplotlib import pyplot as plt 
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, issparse
from scipy.stats import entropy
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TadeClustering():
    # TODO long vector, PCA to 200, t-SNE to 2
    # befe1rko3zo3
    def main(self, print_cutoff=False):
        tade_d = self.read_tade_dict()
        if print_cutoff:
            return self.read_tade_dict(print_cutoff=True)
        self.get_tade_mx(tade_d)
        self.dim_reduce()
        self.plot_clust(self.mx)

    def read_tade_dict(self, print_cutoff=False, short=True):
        logging.info('Reading Tade matrix..')
        tade_d = defaultdict(int)
        self.frame_ent = dict()
        frame_distri = []
        with open('/home/hlt/Language/Hungarian/Dic/tade-cutoff.tsv') as tade_f:
            verb2 = ''
            for line in tade_f:
                verb, frame, freq, vfreq, _ = line.split()
                if print_cutoff and '_' in verb:
                    continue
                freq = int(freq)
                if print_cutoff:
                    if int(vfreq) < 2**(print_cutoff[verb] + 1) * freq:
                        print(line.strip())
                    if priverb != verb2:
                        self.frame_ent[verb2] = entropy(frame_distri)
                        #logging.debug((verb2, entropy(frame_distri)))
                        frame_distri = []
                        verb2 = verb
                    frame_distri.append(freq)
                if short:
                    for cas in frame.split('_'):
                        tade_d[verb, cas] += freq
                else:
                    tade_d[verb, frame] += freq 
            self.frame_ent[verb2] = entropy(frame_distri)
        return tade_d 

    def get_tade_mx(self, tade_d):
        self.verbs, columns = zip(*tade_d.keys())
        column_i = {col: i for i, col in enumerate(set(columns))}
        verb_i = {vrb: i for i, vrb in enumerate(set(self.verbs))}
        #self.mx =  lil_matrix((len(verb_i), len(column_i)))#, 'int')
        self.mx =  np.zeros((len(verb_i), len(column_i)), 'int')
        logging.info(self.mx.shape)
        for vrb, col in  tade_d.keys():
            self.mx[verb_i[vrb], column_i[col]] = tade_d[vrb, col] 

    def dim_reduce(self):
        if self.mx.shape[1] > 500:
            logging.info('PCA..')
            pca = PCA(n_components=200)
            self.mx = pca.fit_transform(self.mx)
        logging.info('t-SNE..')
        tsne = TSNE(init='pca')
        try:
            self.mx = tsne.fit_transform(self.mx)
        except Exception as e:
            logging.error(e)
            raise e
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
        ax.scatter(*zip(*mx))
        for label, row in zip(self.verbs, self.mx):
            plt.annotate(label, xy = row)
        ax.set_aspect('auto')
        plt.show()


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    TadeClustering().main()
