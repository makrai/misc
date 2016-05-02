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
    def __init__(self, short=False, transpose=False, load=False): 
        self.input_filen = '/mnt/store/hlt/Language/Hungarian/Dic/tade/tade_cutoff_with-aux.tsv'
        self.mx_filen = 'tade_mx.npz'
        self.short = short
        self.transpose=transpose
        self.load=load

    def main(self, print_cutoff=False, log_mx=False):
        # TODO mut_info and cocluster are incompatible
        # TODO ACC-ot ku2lo2n kezelni?
        if print_cutoff:
            self.read_frame_ent()
            self.print_freq_frame()
        else: 
            self.read_tade_dict(row_is_prev=True)
            if self.transpose:
                logging.info('Transposing')
                self.mx = self.mx.T
            self.sort_lines(cut_off=(400,1000))
            self.mut_info(log_mx)
            #self.cocluster()
            self.dim_reduce()#apply_tsne=False)
            if self.load:
                np.savez(self.mx_filen, self.mx, self.cols, self.rows)
            self.plot_tade(log_mx)

    def read_tade_dict(self, collate_aux=True, row_is_prev=False):
        logging.info("Reading Tade to a dictionary..")
        # TODO DEBUG load
        if self.load and isfile(self.mx_filen):
            logging.info('Loading mx..')
            self.mx, self.cols, self.rows = np.load(self.mx_filen)
            logging.debug(type(self.mx))
            return
        with open(self.input_filen) as tade_f:
            self.tade_dict = defaultdict(int)
            for line in tade_f:
                verb, frame, freq, vfreq, _ = line.split()
                frame = frame.upper()
                freq = int(freq)
                if collate_aux: 
                    verb = verb.split('_')[-1]
                if row_is_prev:
                    if '+' in verb: 
                        prev, verb = verb.split('+', 1)
                        if '+' in verb:
                            continue 
                        self.book_clause(prev, frame, freq)
                else:
                    self.book_clause(verb, frame, freq)
        self.dict_to_mx()

    def book_clause(self, verb, frame, freq, type_freq=False):
        if type_freq:
            freq = 1
        if self.short:
            for cas in frame.split('_'):
                self.tade_dict[verb, cas] += freq
        else:
            self.tade_dict[verb, frame] += freq 

    def dict_to_mx(self, sparse=False):
        self.rows, self.cols = [np.array(list(set(tuple_)))
                                for tuple_ in zip(*self.tade_dict.keys())]
        logging.debug(self.rows[:7])
        logging.debug(self.cols[:1])
        case_i = {cas: i for i, cas in enumerate(self.cols)}
        verb_i = {vrb: i for i, vrb in enumerate(self.rows)}
        if sparse:
            self.mx =  lil_matrix((len(verb_i), len(case_i)))#, 'int')
        else:
            self.mx =  np.zeros((len(verb_i), len(case_i)))
        logging.info(self.mx.shape)
        for vrb, col in  self.tade_dict.keys():
            self.mx[verb_i[vrb], case_i[col]] = self.tade_dict[vrb, col] 

    def mut_info(self, log_mx):
        logging.info('Computing mutual information..')
        self.mx += 1
        n_token =  self.mx.sum()
        verb_freq = self.mx.sum(axis=1).reshape(-1,1)
        cas_freq = self.mx.sum(axis=0).reshape(1,-1)
        logging.debug('Computing relative freq..')
        self.mx /= verb_freq
        logging.debug('Computing relative freq..')
        self.mx /= cas_freq
        self.mx *= n_token
        if log_mx:
            self.mx = np.log(self.mx)

    def sort_lines(self, cut_off=(-1,-1)):
        verb_argrank = self.mx.sum(axis=1).argsort()[::-1]
        cas_argrank = self.mx.sum(axis=0).argsort()[::-1]
        verb_argrank = verb_argrank[:cut_off[0]]
        cas_argrank = cas_argrank[:cut_off[1]]
        verb_argrank = verb_argrank.reshape(-1,1)
        cas_argrank = cas_argrank.reshape(1,-1) 
        self.mx = self.mx[verb_argrank, cas_argrank]
        self.rows = self.rows[verb_argrank].reshape(-1)
        self.cols = self.cols[cas_argrank].reshape(-1)
        logging.debug(self.rows[:5])
        logging.debug(self.cols[:4])

    def cocluster(self, blockdiag=False):
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
        self.rows = self.rows[np.argsort(clusser.row_labels_)]
        logging.info('Argsorting mx cases..')
        self.mx = self.mx[:, np.argsort(clusser.column_labels_)]
        self.cols = self.cols[np.argsort(clusser.column_labels_)]

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
        logging.debug(self.mx.shape)

    def plot_tade(self, log_mx, savefig=False, row_freq_ent=False, mut_info=False):
        logging.info('Plotting..')
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        if issparse(self.mx): 
            self.ax.spy(self.mx[:200,:200], marker='.')
            self.ax.set_aspect('auto')
        elif row_freq_ent:
            row_freq = self.mx.sum(axis=1).reshape(-1,1)
            row_ent = np.apply_along_axis(entropy, 1, self.mx)
            self.scatter(row_freq, row_ent)
        elif self.mx.shape[1] > 2:
            points, features = self.rows, self.cols
            if self.transpose:
                points, features = features, points
            if log_mx:
                cax = self.ax.matshow(self.mx)
            else:
                cax = self.ax.matshow(self.mx, norm=LogNorm())
            label_limit = 150
            if self.mx.shape[0] < label_limit:
                self.ax.set_yticklabels([''] + self.rows.tolist())
                self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            if self.mx.shape[1] < label_limit:
                self.ax.set_xticklabels([''] + self.cols.tolist())
                self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                locs, labels = plt.xticks()
                plt.setp(labels, rotation=90)
            fig.colorbar(cax)
            self.ax.set_aspect('auto')
        else:
            self.scatter(self.mx[:,0], self.mx[:,1])
        if savefig:
            logging.info('Saving fig..')
            plt.savefig('tade-mutinfo-tsne.pdf')
        else:
            plt.show() 

    def scatter(self, xs, ys):
        self.ax.scatter(xs, ys)
        points, features = self.rows, self.cols
        if self.transpose:
            points, features = features, points
        labels = np.array(['{} {}'.format(l, features[m]) for l, m in zip(
            points, self.mx.argmax(axis=1))])
        for label, row in zip(labels.reshape(-1), zip(xs, ys)):
            break
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
