__author__ = 'Nick Hirakawa'

import re
import sys
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


class CorpusParser:

    def __init__(self, filename):
        self.filename = filename
        self.regex = re.compile('^#\s*\d+')
        self.corpus = dict()

    def parse(self, output):
       with open(self.filename) as f, open(output, 'w+') as o:
            s = ''.join(f.readlines())
            blobs = s.split('\n')[1:]
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                text = x.split()
                text.pop(0)
                docid = text.pop(0)
                stop = stopwords.words('english')
                text = [word for word in ' '.join(text).lower().split() if word not in stop and not word == '.']
                text[-1] = text[-1][:-1]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                text = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]
                o.write("%s %s \n" % (docid, ' '.join(text)))  
                
    def parseComplete(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines())
            blobs = s.split('\n')[1:]
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                text = x.split()
                text.pop(0)
                docid = text.pop(0)
                stop = stopwords.words('english')
                text = [word for word in ' '.join(text).lower().split() if word not in stop]
                text[-1] = text[-1][:-1]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                self.corpus[docid] = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                    
    def readparsed(self):
          with open(self.filename) as f:
            s = ''.join(f.readlines())
            blobs = s.split('\n')[1:]
            for x in blobs:
                text = x.split()
                docid = text.pop(0)
                self.corpus[docid] = text       

    def get_corpus(self):
        return self.corpus


class QueryParser:

    def __init__(self, filename):
        self.filename = filename
        self.queries = []
        self.truths = []

    def parse(self, output):
        with open(self.filename) as f, open(output, 'w+') as o:
            blobs = ''.join(f.readlines()).split('\n')
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                truth = x.split('\t')[-2]
                stop = stopwords.words('english')
                text = [word for word in x.split('\t')[-1].lower().split() if word not in stop]
                text[-1] = text[-1][:-1]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                query = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]
                o.write("%s %s\n" % (truth, ' '.join(query)))
            
  
    def parseComplete(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            lmtzr = WordNetLemmatizer()
            queries = [y[-1] for x in lines.split('\n') for y in [x.split('\t')]]
            for query in queries:
                print("lemmatizing query: " + query)
                stop = stopwords.words('english')                
                query = [word for word in query.lower().split() if word not in stop]
                tokenized = nltk.pos_tag(nltk.word_tokenize(query))
                self.queries.append([lmtzr.lemmatize(WordParser.parseword(query[position]), WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(query))])
            self.truths = [y[-2] for x in lines.split('\n') for y in [x.split('\t')]]
            
    def readparsed(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            self.queries = [y[1:] for x in lines.split('\n') for y in [x.split()]]
            self.truths = [y[0] for x in lines.split('\n') for y in [x.split()]]
            
    def get_queries(self):
        return self.queries

class WordParser:
    
    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.VERB
            
    @staticmethod  
    def parseword(word):
        word = word.lower()
        if word[-1] == ".":
            word = word[:-1]
        return word

if __name__ == '__main__':
    qp = QueryParser('text/queries.txt')
    print(qp.get_queries())