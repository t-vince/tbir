from nltk.corpus import wordnet

class WordParser:
    
    @staticmethod
    def get_wordnet_pos(self, treebank_tag):
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
    def parseword(self, word):
        word = word.lower()
        if word[-1] == ".":
            word = word[:-1]
        return word
