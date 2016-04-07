from collections import defaultdict
import logging

from matplotlib import pyplot as plt 
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.cluster.bicluster import SpectralCoclustering



def read_tade_mx():
    logging.info('Reading Tade matrix..')
    tade_d = dict()
    with open('/mnt/store/hlt/Language/Hungarian/Dic/tade.tsv') as tade_f:
        for line in tade_f:
            verb, frame, freq, _, _ = line.split()
            tade_d[verb, frame] = freq
    verbs, frames = zip(*tade_d.iterkeys())
    frame_i = {frm: i for i, frm in enumerate(set(frames))}
    verb_i = {vrb: i for i, vrb in enumerate(set(verbs))}
    mx =  lil_matrix((len(verb_i), len(frame_i)))#, 'int')
    #mx = csc_matrix(mx)
    for vrb, frm in  tade_d.iterkeys():
        mx[verb_i[vrb], frame_i[frm]] = int(tade_d[vrb, frm])
    return mx


def cocluster(mx):
    logging.info('Co-clustering Tade..')
    clusser = SpectralCoclustering(n_jobs=-1)
    #n_clusters=3, svd_method='randomized',
    clusser.fit(mx)
    fit_data = mx
    """
    logging.info('Argsorting mx..')
    fit_data = mx[np.argsort(clusser.row_labels_)]
    fit_data = fit_data[:, np.argsort(clusser.column_labels_)]
    """
    logging.info(type(fit_data))
    logging.info(fit_data.shape[:2])
    plt.matshow(fit_data[:99,:99].todense())#.tocoo(), cmap=plt.cm.Blues)
    logging.info('')
    plt.show()


if __name__ == '__main__':
    format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_)
    mx = read_tade_mx()
    cocluster(mx)
