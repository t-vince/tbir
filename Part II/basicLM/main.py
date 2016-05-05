from parse import *
from query import *
import operator
import os
import subprocess
import sys
import nltk
import time
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    start = time.time()
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
    smoothparams = [100]

    #step 1: build inverted index
    print('building data structures')
    idx, ft, dlt = build_data_structures(corpus)
    #idx.to_db()
    idx.write("./imagedefault.idx")

    #step 2: run queries against inverted index file
    proc = QueryProcessor(queries, idx = "./imagedefault.idx", dlt=dlt, ft=ft, score_function='Query Likelihood')
    print('running queries')
    for smoothing in smoothparams:
        proc.restart() # start over from the first query
        print('running queries with %d smoothing' % smoothing)
        cut = 1000
        #results = proc.run(smoothing, cut)
        precrec = []
        images = cpi.get_images()
        #for index, result in enumerate(results):
        while proc.hasNext():
            index = proc.getCurrentQueryId()
            result = proc.runNext(smoothing, cut)
            correct = 0
            truth = qpi.truths[index]
            print(truth)
            total = images.count(truth)
            print(total)
            precision = 0
            for ind, ranking in enumerate(result):
                imgid = images[int(ranking[0])]
                if imgid == truth:
                    correct +=1
                    precision += round(correct/(ind+1),10)
            recall = round(correct/total, 10)
            precrec.append((round(precision/cut, 10), recall))
        totalprec = 0
        totalrec = 0
        filename = './imageresults/results_%d.txt' % smoothing
        with open(filename, 'w') as f:
            for prec, rec in precrec:
                f.write(str(prec) + " " + str(rec) + "\n")
                totalprec += prec
                totalrec += rec
        filename = './imageresults/run_%d.txt' % smoothing
        with open(filename, 'w') as f:
            f.write(str(totalprec/len(queries))+"\n")
            f.write(str(totalrec/len(queries)))
    end = time.time()
    print(end - start)


def make_dir():
    if not os.path.exists('./imageresults'):
        os.makedirs('./imageresults')


if __name__ == '__main__':
    make_dir()
    main()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
