#!/usr/bin/env python

import argparse
import codecs
import re
import xml.parsers.expat


class OldLongmanParser:
    def __init__(self, args):
        self.infilen = args.ldoce_xml
        self.outfilen = args.output_tsv
        self.parser = xml.parsers.expat.ParserCreate("utf-8")
        self.parser.StartElementHandler = self.start_element
        self.parser.EndElementHandler = self.end_element
        self.parser.CharacterDataHandler = self.character_data
        self.path_in_xml_tree = []
        self.hwd = ''
        self.lexunit = ''
        self.nonDV = False
        self.write_nonDV = args.write_nonDV
        self.non_dv = ['NonDV', 'FULLFORM']

    def main(self):
        with open(self.infilen) as infile, codecs.open(
                self.outfilen, mode='w', encoding='utf-8') as self.outfile:
            self.parser.ParseFile(infile)

    def start_element(self, name, attrs):
        self.path_in_xml_tree.append(name)
        if name == 'Sense':
            self.lexunit = ''
        elif name == 'HWD':
            self.hwd = ''
        elif name == 'DEF':
            self.defn = ''
            self.outfile.write(self.lexunit if self.lexunit else self.hwd)
            self.outfile.write('\t')
        elif name in self.non_dv:
            if self.write_nonDV:
                self.nonDV = True
                self.defn += ' <{}>'.format(name)

    def end_element(self, name):
        self.path_in_xml_tree.pop()
        if name == 'DEF':
            self.outfile.write(re.sub('\s\s+', ' ', self.defn.strip()))
            self.outfile.write('\n')
        elif name in self.non_dv:
            if self.write_nonDV:
                self.nonDV = False
                self.defn += ' </{}>'.format(name)

    def character_data(self, data):
        if self.path_in_xml_tree[-1] in ['TEXT', 'REFHWD']:
            name = self.path_in_xml_tree[-2]
        else:
            name = self.path_in_xml_tree[-1]
        if data.strip():
            if name == 'HWD':
                self.hwd += data.strip()
            elif name == 'LEXUNIT':
                self.lexunit += data.strip()
            elif (name in ['DEF', 'GLOSS'] or 
                  (self.write_nonDV and name in self.non_dv)):
                self.defn += data


def parse_args():
    parser = argparse.ArgumentParser(description='Parse the Longman dictionary.')
    parser.add_argument('ldoce_xml')
    parser.add_argument('output_tsv')
    parser.add_argument('--write_nonDV', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    OldLongmanParser(parse_args()).main()
