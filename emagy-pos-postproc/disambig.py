
# coding: utf-8

# In[ ]:

from collections import defaultdict
import glob
import gzip
import json
import operator
import os

import logging
logging.basicConfig(
    format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
    level=logging.DEBUG)


# In[ ]:

def zwrite_sorted(freq_dict, outfilen):
    with gzip.open(outfilen, mode='w') as outfile:
        for key, freq in sorted(freq_dict, key=operator.itemgetter(1), reverse=True):
            outfile.write('{}\t{}\n'.format(freq, key))


# In[1]:

def main():
    form_freq = defaultdict(int)
    lemma_freq = defaultdict(int)
    for filen in os.listdir('..'):
        if os.path.exists(filen):
            logging.info('Skipping {}'.format(filen))
            continue
        logging.info('Writing {}'.format(filen))
        with gzip.open(os.path.join('..', filen)) as infile, gzip.open(filen, mode='w') as outfile:
            for line in infile:
                if not line.strip():
                    outfile.write(line)
                    continue
                else:
                    vals = line.decode('utf-8').strip().split('\t')
                try:
                    form, lemma, feats, lemma_feats_ana_json = vals
                except:
                    logging.warning('vals')
                    continue
                ana = []
                if feats == 'OTHER':
                    ana = [feats]
                else:
                    senses = json.loads(lemma_feats_ana_json)
                    for sense_d in  senses:
                        if sense_d['feats'] == feats:
                            ana.append(sense_d['ana'])
                if not ana:
                    # Marmokannájuk
                    pass
                    #logging.warn(vals)
                disamb_vals = (form, lemma, feats, ','.join(sorted(ana)))
                form_freq[form] += 1
                lemma_freq[disamb_vals] += 1
                outfile.write('{}\t{}\t{}\t{}\n'.format(*disamb_vals).encode('utf8'))
    zwrite_sorted(form_freq, 'form_freq.gz')
    zwrite_sorted(lemma_freq, 'lemma_freq.gz')

