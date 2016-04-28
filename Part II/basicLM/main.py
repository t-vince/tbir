from parse import *
from query import *
import operator
import os
import subprocess
import sys
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    qpi = QueryParser(filename='./queries_val_parsed.txt')
    cpi = CorpusParser(filename='./target_collection_parsed.txt')
    qp = QueryParser(filename='./queries.txt')
    cp = CorpusParser(filename='./corpus.txt')
    #cp.parse()
    print('parsing corpus')
    cpi.readparsed()
    #qp.parse()
    print('parsing queries')
    qpi.readparsed()
    queries = qpi.get_queries()
    corpus = cpi.get_corpus()

    #step 1: build inverted index
    print('building data structures')
    idx, ft, dlt = build_data_structures(corpus)
    #idx.to_db()
    idx.write("./imagedefault.idx")

    #step 2: run queries against inverted index file
    proc = QueryProcessor(queries, idx = "./imagedefault.idx", dlt=dlt, ft=ft, score_function='Query Likelihood')
    print('running queries')
    results = proc.run()
    precrec = []
    images = cpi.get_images()
    cut = 1000
    for index, result in enumerate(results):
        s = sorted([(k, v) for k, v in result.items()], key=operator.itemgetter(1))
        s.reverse()
        correct = 0
        cutoff = enumerate(s[:cut])
        truth = qpi.truths[index]
        print(truth)
        total = images.count(truth)
        print(total)
        for ind, ranking in cutoff:
           imgid = images[int(ranking[0])]
           if imgid == truth:
               correct +=1
        precision = round(correct/cut,10)
        recall = round(correct/total, 10)
        precrec.append((precision, recall))
    totalprec = 0
    totalrec = 0
    filename = './imageresults/results.txt'
    with open(filename, 'w') as f:
        for prec, rec in precrec:
            f.write(str(prec) + " " + str(rec) + "\n")
            totalprec += prec
            totalrec += rec
    filename = './imageresults/run.txt'
    with open(filename, 'w') as f:
        f.write(str(totalprec/len(results))+"\n")
        f.write(str(totalrec/len(results)))


def make_dir():
    if not os.path.exists('./imageresults'):
        os.makedirs('./imageresults')


if __name__ == '__main__':
    make_dir()
    main()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
