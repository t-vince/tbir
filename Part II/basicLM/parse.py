__author__ = 'Nick Hirakawa'

import re
import sys
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob


class CorpusParser:

    def __init__(self, filename):
        self.filename = filename
        self.regex = re.compile('^#\s*\d+')
        self.corpus = dict()
        self.images = []

    def parse(self, output):
       with open(self.filename) as f, open(output, 'w+') as o:
            s = ''.join(f.readlines())
            blobs = s.split('\n')[1:]
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                text = x.split()
                docid = text.pop(0)
                imgid = text.pop(0)
                text = ' '.join(text).lower().translate(dict.fromkeys(map(ord, u".,:;\"'")))
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop and not word == '.']
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                text = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                o.write("%s %s %s \n" % (docid, imgid, ' '.join(text)))
                
    def parseComplete(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines())
            blobs = s.split('\n')
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                text = x.split()
                docid = text.pop(0)
                imgid = text.pop(0)
                text = ' '.join(text).lower().translate(dict.fromkeys(map(ord, u".,:;\"'")))
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                self.corpus[docid] = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                self.images[imgid] = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                    
    def readparsed(self):
          with open(self.filename) as f:
              lines = ''.join(f.readlines())
              self.images.append("")
              for x in lines.split('\n'):
                  for y in [x.split()]:
                      self.images.append(y[1])
              self.corpus = {y[0]:y[2:] for x in lines.split('\n') for y in [x.split()]}

    def get_corpus(self):
        return self.corpus
        
    def get_images(self):
        return self.images


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
                text = x.split('\t')[-1].lower()
                truth = x.split('\t')[-2]
                '''
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop]
                text[-1] = text[-1][:-1]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                query = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]
                ''' 
                stemmer = SnowballStemmer("english", ignore_stopwords=True)
                text = [stemmer.stem(word) for word in text.split()]
                o.write("%s %s\n" % (truth, ' '.join(text)))
            
  
    def parseComplete(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            lmtzr = WordNetLemmatizer()
            queries = [y[-1] for x in lines.split('\n') for y in [x.split('\t')]]
            for query in queries:
                query = query.lower()
                stop = stopwords.words('english')                
                query = [word for word in query.split() if word not in stop]
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