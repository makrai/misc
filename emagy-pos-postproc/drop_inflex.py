
# coding: utf-8

# In[45]:

from collections import defaultdict
import glob
import gzip
import operator
import os
import re

import logging
logging.basicConfig(
    format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
    level=logging.DEBUG)

from disambig import zwrite_sorted


# In[64]:

def zwrite_sorted(freq_dict, outfilen):
    with gzip.open(outfilen, mode='w') as outfile:
        for (lemma, pos, anal), freq in sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True):
            outfile.write('{}\t{}\t{}\t{}\n'.format(freq, lemma, pos, anal).encode('utf-8'))


# In[65]:

proj_dir = '/mnt/permanent/home/makrai/data/tmp_webkorp_emagy_pos_without_inflex/'
freq = defaultdict(int)
for infilen in glob.glob('/mnt/permanent/Language/Hungarian/Crawl/Web2/emagyar_pos/disamb/web2-4p-*.gz'):
    logging.info(infilen)
    with gzip.open(infilen) as infile,             gzip.open(os.path.join(proj_dir, os.path.basename(infilen)), mode='w') as outfile:
        for line in infile:
            if not line.strip():
                #outfile.write(line)
                continue
            vals = line.strip().decode('utf-8').split(sep='\t')
            if len(vals) ==3:
                # Marmokannájuk
                # outfile.write()
                continue
            form, lemma, pos, ana = vals
            if pos == 'OTHER':
                continue
            if re.search('[,+]', form):
                # e.g. 8,5
                continue
            bracketed = '\[[^\]]*\]'
            pos = re.findall(bracketed, pos)[0]
            deriv_anals = []
            for disamb_anal in ana.split(','):
                deriv_anal = []
                for segment in disamb_anal.split('+'):
                    if '=' in segment:
                        # e.g. ('állít[/V]', 'állít')
                        deep, surf = segment.rsplit('=', 1) 
                    else:
                        # bug?, e.g. ők[/N|Pro]
                        deep = segment
                        surf = segment
                    deep_cat = re.findall(bracketed, deep)[0]
                    if '/' in deep_cat:
                        deriv_anal.append(segment)
                deriv_anals.append('+'.join(sorted(set(deriv_anal))))
            freq[(lemma, pos, ','.join(sorted(set(deriv_anals))))] += 1       
zwrite_sorted(freq, 'freq.ztsv')            

