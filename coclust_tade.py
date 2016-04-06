from collections import defaultdict

from matplotlib import pyplot as plt 
from scipy.sparse import csc_matrix
from sklearn.cluster.bicluster import SpectralCoclustering



def read_tade_mx():
    tade_d = defaultdict(dict)
    frames = set()
    with open('/mnt/store/hlt/Language/Hungarian/Dic/tade.tsv') as tade_f:
        for line in tade:
            verb, frame, freq, _, _ = line.split()
            tade_d[verb, frame] = freq
            frames.add(frame)
    frame_i = {frm: i for i, frm in enumerate(frames)}
    verb_i = {vrb: i for i, vrb in enumerate(tade_d)}
    mx =  csc_matrix((len(v_i), len(frame_i)), int)
    for vrb, frms in  tade_d:
        for frm, frq in frms:
            mx[verb_i[vrb]][frame_i[frm]] = frq
    return mx

def cocluster(mx):
    clusser = SpectralCoclustering(n_jobs=-1)
    #n_clusters=3, svd_method='randomized',
    clusser.fit(mx)
    fit_data = data[np.argsort(clusser.row_labels_)]
    fit_data = fit_data[:, np.argsort(clusser.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.show()

if __name__ == '__main()__':
    mx = read_tade_mx()
    cocluster(mx)
