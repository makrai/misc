from collections import defaultdict
import logging
from os.path import isfile

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, issparse
from scipy.stats import entropy
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm
from matplotlib import ticker


class TadeClustering(): 
    def __init__(self, short=True, log_mx=False, sum_axis=1):
        self.input_filen = '/mnt/store/hlt/Language/Hungarian/Dic/tade/tade_cutoff_with-aux.tsv'
        self.short = short
        self.log_mx = log_mx
        self.sum_axis = sum_axis

    def main(self, print_cutoff=False):
        if print_cutoff:
            self.read_frame_ent()
            self.print_freq_frame()
        else: 
            tade_dict = self.read_tade_dict(type_freq=True)
            freq_mx = self.dict_to_mx(tade_dict).sum(axis=self.sum_axis) 
            freq_mx = self.sort_lines(freq_mx, cut_off=(50,100))#400,1000))
            mi_mx = self.mut_info(np.copy(freq_mx))
            #self.cond_depend()
            #self.cocluster()#blockdiag=True)
            #self.dim_reduce()#apply_tsne=False)
            self.plot_tade(freq_mx, mi_mx)#, fig_filen='tade-prev-cas-token.pdf')

    def read_tade_dict(self, type_freq=True, collate_aux=True):
        logging.info("Reading Tade to a dictionary..")
        with open(self.input_filen) as tade_f:
            tade_dict = defaultdict(int)
            for line in tade_f:
                verb, frame, freq, vfreq, _ = line.split()
                frame = frame.upper() 
                freq = 1 if type_freq else int(freq)
                if collate_aux: 
                    verb = verb.split('_')[-1]
                if '+' in verb: 
                    prev, stem = verb.split('+', 1)
                    if '+' in stem:
                        continue 
                else:
                    prev, stem = '@', verb
                if self.short: 
                    for cas in frame.split('_'):
                        tade_dict[prev, stem, cas] += freq
                else:
                    tade_dict[prev, stem, frame] += freq 
        return tade_dict

    def dict_to_mx(self, tade_dict, sparse=False):
        self.prev, self.stems, self.case = [
            np.array(list(set(tuple_)))
            for tuple_ in zip(*tade_dict.keys())]
        logging.debug(self.prev[:7])
        logging.debug(self.stems[:7])
        logging.debug(self.case[:7])
        prev_i = {key: i for i, key in enumerate(self.prev)}
        stem_i = {key: i for i, key in enumerate(self.stems)}
        col_i = {key: i for i, key in enumerate(self.case)}
        shape = (len(prev_i), len(stem_i), len(col_i))
        logging.info(shape)
        mx =  lil_matrix(shape) if sparse else np.zeros(shape)
        for prev, stem, col in  tade_dict:
            mx[prev_i[prev], stem_i[stem], col_i[col]] = tade_dict[
                                                              prev, stem, col]
        return mx

    def sort_lines(self, mx, cut_off=(-1,-1)):
        logging.info("Sorting array..")
        row_argrank = mx.sum(axis=1).argsort()[::-1]
        col_argrank = mx.sum(axis=0).argsort()[::-1]
        row_argrank = row_argrank[:cut_off[0]]
        col_argrank = col_argrank[:cut_off[1]]
        row_argrank = row_argrank.reshape(-1,1)
        col_argrank = col_argrank.reshape(1,-1) 
        mx = mx[row_argrank, col_argrank]
        if self.sum_axis == 0:
            self.rows = self.stems[row_argrank].reshape(-1)
            self.cols = self.case[col_argrank].reshape(-1)
        if self.sum_axis == 1:
            self.rows = self.prev[row_argrank].reshape(-1)
            self.cols = self.case[col_argrank].reshape(-1)
        if self.sum_axis == 2:
            self.rows = self.prev[row_argrank].reshape(-1)
            self.cols = self.stems[col_argrank].reshape(-1)
        logging.debug(self.rows[:5])
        logging.debug(self.cols[:5])
        return mx

    def mut_info(self, mx):
        logging.info('Computing mutual information..')
        mx += 1
        n_token =  mx.sum()
        row_sum = mx.sum(axis=1).reshape(-1,1)
        col_sum = mx.sum(axis=0).reshape(1,-1)
        logging.debug('Computing relative freq..')
        mx /= row_sum
        logging.debug('Computing relative freq..')
        mx /= col_sum
        mx *= n_token
        if self.log_mx:
            mx = np.log(mx)
        return mx

    def cond_depend(self, probab_axis=2):
        logging.info("Computing conditional dependency..")
        shape = list(mx.shape)
        shape[probab_axis] = -1
        shape = tuple(shape)
        mx += 1
        mx /= mx.sum(axis=probab_axis).reshape(shape)
        mx = mx.sum(axis=self.sum_axis) 

    def cocluster(self, blockdiag=False):
        logging.info('Co-clustering Tade..')
        if blockdiag:
            logging.info('blockdiag')
            clusser = SpectralCoclustering(n_jobs=-1)
        else: # checkerboard
            logging.info('checkerboard')
            clusser = SpectralBiclustering(n_jobs=-1, n_clusters=(4,3))
            #n_clusters=3, svd_method='randomized',
        clusser.fit(mx)
        logging.info('Argsorting mx rows..')
        mx = mx[np.argsort(clusser.row_labels_)]
        self.prev = self.prev[np.argsort(clusser.row_labels_)]
        logging.info('Argsorting mx cases..')
        mx = mx[:, np.argsort(clusser.column_labels_)]
        self.case = self.case[np.argsort(clusser.column_labels_)]

    def dim_reduce(self, apply_tsne=True):
        if mx.shape[1] > 400 or not apply_tsne:
            logging.info('PCA.. from {}'.format(mx.shape))
            pca = PCA(n_components=200 if apply_tsne else 2)
            mx = pca.fit_transform(mx)
        if apply_tsne:
            logging.info('t-SNE.. from {}'.format(mx.shape))
            tsne = TSNE(init='pca')
            mx = tsne.fit_transform(mx)
        # method : string (default: 'barnes_hut') 
        #   By default the gradient calculation algorithm uses Barnes-Hut
        #   approximation running in O(NlogN) time. method='exact' will run on
        #   the slower, but exact, algorithm in O(N^2) time
        logging.debug(mx.shape)

    def plot_tade(self, freq_mx, mi_mx, fig_filen=None, row_freq_ent=False):
        logging.info('Plotting..')
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        if issparse(mi_mx): 
            self.ax.spy(mi_mx[:200,:200], marker='.')
            self.ax.set_aspect('auto')
        elif row_freq_ent:
            row_freq = mi_mx.sum(axis=1).reshape(-1,1)
            row_ent = np.apply_along_axis(entropy, 1, mi_mx)
            self.scatter(row_freq, row_ent)
        elif mi_mx.shape[1] > 2: # matshow
            if self.log_mx:
                cax = self.ax.matshow(mi_mx)
            else:
                cax = self.ax.matshow(np.where(freq_mx>2, mi_mx,
                                               np.zeros(mi_mx.shape)), norm=LogNorm())
            label_limit = 100
            if mi_mx.shape[0] <= label_limit:
                self.ax.set_yticklabels([''] + self.rows.tolist())
                self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            if mi_mx.shape[1] <= label_limit:
                cas_proper = lambda col: col.split('<')[-1] if '<' in col else col
                self.ax.set_xticklabels([''] + list(map(cas_proper,
                                                        self.cols)))
                self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                locs, labels = plt.xticks()
                plt.setp(labels, rotation=90)
            fig.colorbar(cax)
            self.ax.set_aspect('auto')
        else:
            self.scatter(mi_mx[:,0], mi_mx[:,1])
        if fig_filen:
            logging.info('Saving fig..')
            plt.savefig(fig_filen)
        else:
            plt.show() 

    def scatter(self, xs, ys):
        self.ax.scatter(xs, ys)
        points, features = self.prev, self.case
        labels = np.array(['{} {}'.format(l, features[m]) for l, m in zip(
            points, mx.argmax(axis=1))])
        for label, row in zip(labels.reshape(-1), zip(xs, ys)):
            self.ax.annotate(label, xy=row, fontsize=10)
        #self.ax.set_xscale('log')

    def read_frame_ent(self):
        self.frame_ent = dict()
        with open(self.input_filen) as tade_f:
            last_verb = ''
            frame_distri = []
            for line in tade_f:
                verb, frame, freq, vfreq, _ = line.split()
                freq = int(freq)
                if verb != last_verb:
                    self.frame_ent[last_verb] = entropy(frame_distri)
                    frame_distri = []
                    last_verb = verb
                frame_distri.append(freq)
            self.frame_ent[last_verb] = entropy(frame_distri)

    def print_freq_frame(self, drop_aux=True):
        with open(self.input_filen) as tade_f:
            for line in tade_f: 
                verb, frame, freq, vfreq, _ = line.split()
                freq = int(freq)
                if drop_aux and '_' in verb: 
                    continue
                if int(vfreq) < 2**(self.frame_ent[verb] + 1) * freq: 
                    print(line.strip())


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    TadeClustering().main()
