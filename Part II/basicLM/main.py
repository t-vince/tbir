from parse import *
from query import *
import os
import time
from invdx import *
import sys

def main():
    start = time.time()
    qpi = QueryParser(filename='./queries_val_parsed.txt')
    cpi = CorpusParser(filename='./target_collection_parsed.txt')
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

    #step 2: run queries against inverted index file
    proc = QueryProcessor(queries, idx = "./imagedefault.idx", dlt=dlt, ft=ft, score_function='Query Likelihood')
    print('running queries')  
    cut = 1000
    for smoothing in smoothparams:        
        proc.restart() # start over from the first query
        print("running queries with %d smoothing" % smoothing)
        #results = proc.run(smoothing, cut)
        queryprecrec = []
        images = cpi.get_images()
        #for index, result in enumerate(results):
        while proc.hasNext():
            index = proc.getCurrentQueryId()
            print("running query %d" % index)
            result = proc.runNext(smoothing, cut)
            correct = 0
            truth = qpi.truths[index]
            total = images.count(truth) 
            cutprecrec = []
            singleprecision = 0 
            for c in range(0, cut):
                imgid = images[int(result[c][0])]
                if imgid == truth:
                    correct +=1
                    singleprecision += correct/float(c+1)
                singlerecall = round(correct/total, 10)
                cutprecrec.append((round(singleprecision/cut, 10), singlerecall))            
            queryprecrec.append(cutprecrec)
        precrec = []
        for cutc in range(cut):
            cutprecision = 0
            cutrecall = 0
            for query in queryprecrec:
                cutprecision += query[cutc][0]
                cutrecall += query[cutc][1]
            finalprec = round(cutprecision/cut, 10) 
            finalrec = round(cutrecall/cut, 10) 
            precrec.append((finalprec, finalrec))
        filename = './imageresults/Yresults_%d.txt' % smoothing
        with open(filename, 'w') as f:
            for prec, rec in precrec:
                f.write(str(prec) + " " + str(rec) + "\n")


def make_dir():
    if not os.path.exists('./imageresults'):
        os.makedirs('./imageresults')


if __name__ == '__main__':
    make_dir()
    main()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
