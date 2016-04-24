__author__ = 'Nick Hirakawa'

import re
import sys


class CorpusParser:

    def __init__(self, filename):
        self.filename = filename
        self.regex = re.compile('^#\s*\d+')
        self.corpus = dict()

    def parse(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines())
            blobs = s.split('#')[1:]
            for x in blobs:
                text = x.split()
                docid = text.pop(0)
                self.corpus[docid] = text
                
    def parseImages(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines())
            blobs = s.split('\n')[1:]
            for x in blobs:
                text = x.split()
                docid = text.pop(0)
                text.pop(0)
                self.corpus[docid] = text
              

    def get_corpus(self):
        return self.corpus


class QueryParser:

    def __init__(self, filename):
        self.filename = filename
        self.queries = []

    def parse(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            self.queries = [x.rstrip().split() for x in lines.split('\n')[:-1]]
            
  
    def parseImages(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            self.queries = [y[-1].split() for y in (x.split("\t") for x in lines.split('\n'))]

    def get_queries(self):
        return self.queries


if __name__ == '__main__':
	qp = QueryParser('text/queries.txt')
	print qp.get_queries()