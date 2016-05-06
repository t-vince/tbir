__author__ = 'Nick Hirakawa'

import re
import sys
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob

def clean_input(text):
    # lowecase and remove linebreaks
    text = text.lower().rstrip()
    # Remove punctuation
    text = re.sub('[!@#$:;%&?,_\.\'\`\"\\\/\(\)\[\]]', '', text)
    text = re.sub('[\-]+', '-', text)
    # Remove sole numbers, dashes or extra spaces
    text = re.sub('[\s][\-]+[\s]', '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[\s]+', ' ', text)
    # British to American English - at this moment still hardcoded due to lack of library
    text = text.replace('grey', 'gray')
    text = text.replace('colour', 'color')
    text = text.replace('tyre', 'tire')
    text = text.replace('centre', 'center')
    text = text.replace('theatre', 'theater')
    text = text.replace('jewellery','jewelry')
    text = text.replace('aeroplane', 'plane')
    text = text.replace('harbour', 'harbor')
    text = text.replace('moustache','mustache')
    text = text.replace(' axe', ' hatchet')
    text = text.replace('armour', 'armor')
    text = text.replace('stylised', 'stylized')
    text = text.replace('organise', 'organize')
    text = text.replace('plough', 'plow')
    text = text.replace('neighbourhood', 'neighborhood')
    text = text.replace('vapour', 'vapor')
    # some manual fixes of lemmatizing
    text = text.replace('watersid ', 'waterside ')
    text = text.replace('figur ', 'figure ')
    text = text.replace(' graz ', ' graze ')
    return text

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
                text = clean_input(text)
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop and not word == '.']
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                text = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                o.write("%s %s %s \n" % (docid, imgid, ' '.join(text)))
                
    def parseComplete(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines()[1:])
            blobs = s.split('\n')
            lmtzr = WordNetLemmatizer()
            for x in blobs[:-1]:
                text = x.split()
                print(text)
                docid = int(text.pop(0))
                imgid = text.pop(0)
                text = clean_input(' '.join(text))
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                self.corpus[docid] = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]                
                self.images.append(imgid)               
                    
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
        self.ids = []

    def parse(self, output):
        with open(self.filename) as f, open(output, 'w+') as o:
            blobs = ''.join(f.readlines()).split('\n')
            lmtzr = WordNetLemmatizer()
            for x in blobs:
                sent_id = x.split('t')[0]
                text = x.split('\t')[-1].lower()
                truth = x.split('\t')[-2]
                stop = stopwords.words('english')
                text = [word for word in text.split() if word not in stop]
                text[-1] = text[-1][:-1]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(text)))
                text = [lmtzr.lemmatize(text[position], WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(text))]
                o.write("%s %s\n" % (truth, ' '.join(text)))
            
  
    def parseComplete(self, truth):
        with open(self.filename) as f:
            lines = f.readlines()[1:]
            lines = ''.join(lines)
            lmtzr = WordNetLemmatizer()
            queries = [y[-1] for x in lines.split('\n') for y in [x.split('\t')]]
            for query in queries:
                stop = stopwords.words('english')   
                query = clean_input(query)
                query = [word for word in query.split() if word not in stop]
                tokenized = nltk.pos_tag(nltk.word_tokenize(' '.join(query)))
                self.queries.append([lmtzr.lemmatize(WordParser.parseword(query[position]), WordParser.get_wordnet_pos(tokenized[position][1])) for position in range(0, len(query))])
            self.ids = [y[0] for x in lines.split('\n') for y in [x.split('\t')]]
            if truth:
                self.truths = [y[-2] for x in lines.split('\n') for y in [x.split('\t')]]
            else:
                self.truths = []
            
    def readparsed(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
            self.queries = [y[1:] for x in lines.split('\n') for y in [x.split()]]
            self.truths = [y[0] for x in lines.split('\n') for y in [x.split()]]
            
    def get_queries(self):
        return self.queries
        
    def get_ids(self):
        return self.ids

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