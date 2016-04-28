from parse import *
from query import *
import operator
import os
import subprocess
import sys
import nltk
import scipy as sc
import numpy as np

def lsimain():
    qpi = QueryParser(filename='./queries_val1_parsed.txt')
    cpi = CorpusParser(filename='./target_collection1_parsed.txt')
    print('parsing corpus')
    cpi.readparsed()
    #qp.parse()
    print('parsing queries')
    qpi.readparsed()
    corpus = cpi.get_corpus()
    idx, ft, dlt = build_data_structures(corpus)
    dtmatrix = idx.transform(dlt, ft)
    print(dtmatrix)
    u, s, vt = sc.linalg.svd(dtmatrix, full_matrices=False)
    numDimensions = 1
    u = u[:, :numDimensions]
    sigma = sc.diag(s)[:numDimensions, :numDimensions]
    vt = vt[:numDimensions, :]
    lowRankDocumentTermMatrix = np.dot(u, np.dot(sigma, vt))
    print("-----------------")
    print(lowRankDocumentTermMatrix)
    print("-----------------")
    
    queryVector = dtmatrix[:, 0]
    lowDimensionalQuery = np.dot(np.linalg.inv(sigma), np.dot(u.T, queryVector))

    '''
    lsi = LsiModel(cpi, num_topics=10)
    print(lsi[doc_tfidf]) # project some document into LSI space
    lsi.add_documents(corpus2) # update LSI on additional documents
    print(lsi[doc_tfidf])
    '''
def make_dir():
    if not os.path.exists('./lsiresults'):
        os.makedirs('./lsiresults')


if __name__ == '__main__':
    make_dir()
    lsimain()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
# -*- coding: utf-8 -*-

