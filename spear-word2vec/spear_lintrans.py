import simplejson
import logging

from gensim_wrapper import GensimWrapper
from dinu14.train_test_tm import train_test_wrapper

class LinTransWrapper():
    def __init__(self, job_id, params):
        format_="%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
        logging.basicConfig(format=format_, level=logging.DEBUG)
        self.params = params
        self.langs_conf = simplejson.load(open('/home/makrai/project/efnilex/lang.json'))
        self.sr_code = 'en'
        self.tg_code = 'hu'

    def main(self):
        dict_fn = self.langs_conf['biling'][
            '{}-{}'.format(*sorted([self.sr_code, self.tg_code]))][
                self.params['dict_i'][0]]
        source_fn = self.langs_conf['mono'][self.sr_code]['embed'][self.params[
            'embed_i'][0]]
        target_fn = self.get_tg_embed(self.tg_code, self.params)
        dinu_args = {'seed_fn': dict_fn.format(self.tg_code, self.sr_code),
                     'source_fn': source_fn,
                     'target_fn': target_fn,
                     'reverse': True}
        for key in ['additional', 'train_size']:
            dinu_args[key] = self.params[key] 
        return 1 - train_test_wrapper(*dinu_args)

    def get_tg_embed(self, lang_code, params):
        lang_conf = self.langs_conf['mono'][lang_code]
        gensim_args = dict((key, params[key][0]) for key in [
            'window', 'min_count', 'sg', 'hs', 'negative', 'iter'])
        gensim_args['corpus_fn'] = lang_conf['corp'][params['corp_i'][0]]['small_file']
        gensim_args['size'] = params['dim'][0]
        gensim_args['workers'] = 6
        embed_tmplt = '/mnt/store/hlt/Language/{}/Embed/{}/{}' 
        corp_name = lang_conf['corp'][params['corp_i'][0]]['name']
        gensim_args['model_path_and_prefix'] = embed_tmplt.format(
            lang_conf['name'], corp_name, corp_name)
        return GensimWrapper(gensim_args)


def main(job_id, params):
    return LinTransWrapper(job_id, params).main()
