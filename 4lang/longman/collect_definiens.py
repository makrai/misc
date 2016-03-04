#!/usr/bin/env python                                                                                                               
# disregarded: 'COLLOINEXA', 'GLOSS', 'UB', 'UR'

import argparse
import codecs
import sys
import xml.parsers.expat


class OldLongmanParser:

    def __init__(self, args):
        self.infilen = args.ldoce_xml
        self.outfile = args.output_tsv
        self.parser = xml.parsers.expat.ParserCreate( "utf-8" )
        self.parser.StartElementHandler = self.start_element
        self.parser.EndElementHandler = self.end_element
        self.parser.CharacterDataHandler = self.character_data
        self.path_in_xml_tree = []
        self.hwd = ''
        self.lexunit = ''
        self.nonDV = False
        self.include_nonDV = args.include_nonDV

    def main(self):
        with open(self.infilen) as infile:
            print infile.read()
            self.parser.ParseFile(infile)

    def start_element(self, name, attrs):
        self.path_in_xml_tree.append(name)
        if name == 'Sense':
            self.lexunit = ''
        elif name == 'HWD':
            self.hwd = ''
        elif name == 'DEF':
            if self.lexunit:
                self.outfile.write(self.lexunit) 
            else:
                self.outfile.write(self.hwd)
            self.outfile.write('\t')
        elif name == 'NonDV' and self.include_nonDV:
            self.nonDV = True
            self.outfile.write(' <NonDV>')

    def end_element(self, name):
        self.path_in_xml_tree.pop()
        if name == 'DEF':
            self.outfile.write('\n')
        elif name == 'NonDV' and self.include_nonDV:
            self.nonDV = False
            self.outfile.write(' </NonDV>')

    def character_data(self, data):
        if self.path_in_xml_tree[-1] == 'TEXT':
            name = self.path_in_xml_tree[-2]
        else:
            name = self.path_in_xml_tree[-1]
        stripped = data.strip()
        if stripped != '':
            if name == 'HWD':
                self.hwd += stripped
            elif name == 'LEXUNIT': 
                self.lexunit += stripped
            elif  name == 'DEF' or (self.nonDV and self.include_nonDV):
                self.outfile.write(re.sub(' *', ' ', data))

def parse_args():
    parser = argparse.ArgumentParser(description='Parse the Longman dictionary.')
    parser.add_argument('ldoce_xml')
    parser.add_argument('output_tsv') 
    parser.add_argument('--skip_nonDV', action='store_true', dest='include_nonDV')
    return parser.parse_args()

if __name__ == "__main__":
    OldLongmanParser(parse_args()).main()
