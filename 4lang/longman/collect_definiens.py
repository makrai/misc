#!/usr/bin/env python
# disregarded: 'COLLOINEXA', 'GLOSS', 'UB', 'UR'

import argparse
import codecs
import re
import xml.parsers.expat


class OldLongmanParser:
    # TODO REFHWD 
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
        self.include_nonDV = args.include_nonDV

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
            #self.def_segms = [] # TODO def_segms
            self.outfile.write(self.lexunit if self.lexunit else self.hwd)
            self.outfile.write('\t')
        elif name == 'NonDV':
            if self.include_nonDV:
                self.nonDV = True
                self.outfile.write('<NonDV>')
                # TODO self.def_segms.append('<NonDV>')

    def end_element(self, name):
        self.path_in_xml_tree.pop()
        if name == 'DEF':
            self.outfile.write(self.defn.strip()) 
            #self.outfile.write(" ".join(self.def_segms).strip())
            self.outfile.write('\n')
        elif name == 'NonDV' and self.include_nonDV:
            self.nonDV = False
            self.outfile.write(' </NonDV>')
                # TODO self.def_segms.append('</NonDV>')

    def character_data(self, data):
        if self.path_in_xml_tree[-1] == 'TEXT':
            name = self.path_in_xml_tree[-2]
        else:
            name = self.path_in_xml_tree[-1]
        if data.strip():
            if name == 'HWD':
                self.hwd += data.strip()
            elif name == 'LEXUNIT':
                self.lexunit += data.strip()
            elif name == 'DEF' or (self.nonDV and self.include_nonDV):
                self.defn += data
                return 
                if data.startswith(' ') or not self.def_segms:
                    self.def_segms.append(data.strip())
                else:
                    self.def_segms[-1] += data.strip()


def parse_args():
    parser = argparse.ArgumentParser(description='Parse the Longman dictionary.')
    parser.add_argument('ldoce_xml')
    parser.add_argument('output_tsv')
    parser.add_argument('--skip_nonDV', action='store_true', dest='include_nonDV')
    return parser.parse_args()

if __name__ == "__main__":
    OldLongmanParser(parse_args()).main()
