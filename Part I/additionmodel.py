# Import modules
from gensim import models
#import numpy as np
import logging

# setup logging [NOTE: Logging doesnt output in the newest jupyter kernal, bugs not yet fixed]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def analogy_model1(a, b, c, model): 
    result = model.most_similar(positive=[c, b], negative=[a], topn=1)
    return result[0]

def analogy_model2(a, b, c, model):
    result = model.most_similar_cosmul(positive=[c, b], negative=[a], topn=1)
    return result[0]

def recall_analogy_model(questions, analogy_model, model):
    right_count = 0 
    total_count = 0
    skipped_count = 0
    current_cat = ""
    curr_total = 0
    curr_right = 0
    curr_skipped = 0

    with open(questions, 'r') as file:
        for line in file:
            if line[0] != ':' :   # Ignore the lines that start with a ':', they indicate semantic/syntactic relation categories
                total_count += 1
                curr_total += 1
                words = line.split() # Split the different words
                try:
                    result_text = analogy_model(words[0], words[1], words[2], model)                 
                    if result_text[0] == words[3]:
                        right_count += 1
                        curr_right += 1
                except KeyError:
                    skipped_count += 1
                    curr_skipped += 1
            else :
                if curr_total > 0:
                    curr_recall = float(curr_right) / float(curr_total)
                    if curr_total > curr_skipped:
                        curr_recall_c = float(curr_right) / float(curr_total - curr_skipped)
                    else :
                        curr_recall_c = 0
                    print ("Category", current_cat, end=" ")
                    print ("Recall: ", curr_recall, end=" | ")
                    print ("Recall ignore skipped: ", curr_recall_c, end=" | ")
                    print ("#skipped: ", curr_skipped)
                current_cat = line
                curr_total = 0
                curr_right = 0
                curr_skipped = 0

    # Print last category
    curr_recall = float(curr_right) / float(curr_total)
    if curr_total > curr_skipped:
        curr_recall_c = float(curr_right) / float(curr_total - curr_skipped)
    else :
        curr_recall_c = 0
    print ("Category", current_cat, end=" ")
    print ("Recall: ", curr_recall, end=" | ")
    print ("Recall ignore skipped: ", curr_recall_c, end=" | ")
    print ("#skipped: ", curr_skipped)

    # Return totals
    recall = float(right_count) / float(total_count)
    return float('%.5f'% recall), float(right_count), float(total_count), float(skipped_count)

import smart_open
import os.path

def glove2word2vec(glove_filename):
    def get_info(glove_filename): 
        num_lines = sum(1 for line in smart_open.smart_open(glove_filename))
        dims = glove_filename.split('.')[2].split('d')[0] # file name contains the number of dimensions
        return num_lines, dims
    
    def prepend_info(infile, outfile, line): # Function to prepend lines using smart_open
        with open(infile, 'r', encoding="utf8") as original: data = original.read()
        with open(outfile, 'w', encoding="utf8") as modified: modified.write(line + '\n' + data)
        return outfile
    
    word2vec_filename = glove_filename[:-3] + "word2vec.txt"
    if os.path.isfile(word2vec_filename):
        model = models.Word2Vec.load_word2vec_format(word2vec_filename)
    else:
        num_lines, dims = get_info(glove_filename)
        gensim_first_line = "{} {}".format(num_lines, dims)
        model_file = prepend_info(glove_filename, word2vec_filename, gensim_first_line)
        model = models.Word2Vec.load_word2vec_format(model_file)
    
    model.init_sims(replace = True)  # normalize all word vectors
    return model

'''
glove50d_model = glove2word2vec('data/glove.6B.50d.txt')
print (" -- GloVe 50d Model --")
print ("GloVe50d - addition model", recall_analogy_model('data/questions-words.txt', analogy_model1, glove50d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
print ("GloVe50d - multiplication model", recall_analogy_model('data/questions-words.txt', analogy_model2, glove50d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
glove50d_model = None

glove100d_model = glove2word2vec('data/glove.6B.100d.txt')
print (" -- GloVe 100d Model --")
print ("GloVe100d - addition model", recall_analogy_model('data/questions-words.txt', analogy_model1, glove100d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
print ("GloVe100d - multiplication model", recall_analogy_model('data/questions-words.txt', analogy_model2, glove100d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
glove100d_model = None

glove200d_model = glove2word2vec('data/glove.6B.200d.txt')
print (" -- GloVe 200d Model --")
print ("GloVe200d - addition model", recall_analogy_model('data/questions-words.txt', analogy_model1, glove200d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
print ("GloVe200d - multiplication model", recall_analogy_model('data/questions-words.txt', analogy_model2, glove200d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
'''

# print values per model
glove300d_model = glove2word2vec('data/glove.6B.300d.txt')

print (" -- GloVe 300d Model --")
print ("GloVe300d - addition model", recall_analogy_model('data/questions-words.txt', analogy_model1, glove300d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")
print ("GloVe300d - multiplication model", recall_analogy_model('data/questions-words.txt', analogy_model2, glove300d_model))
print ("Totals show recall, right count, total count, skipped count")
print ("")

# Load Googles' pre-trained Word2Vec vector set
# Note: This will take a lot of memory and can take a while.
# Note II: Depending on your RAM, do not load all models at the same time

#w2v_model = models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

#w2v_model.init_sims(replace=True) # Trim unneeded model memory = use (much) less RAM.

#print "Word2Vec - addition model", recall_analogy_model('data/questions-words.txt', analogy_model1, w2v_model)
