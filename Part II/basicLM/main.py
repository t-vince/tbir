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
        cutMaxIncrease = 1000
        
        totalPrecisionAndRecallPerCutIncrease = [[0] * 2 for i in range(cutMaxIncrease+1)] # Create an array of empty arrays
        images = cpi.get_images()
        
        
        while proc.hasNext():
            index = proc.getCurrentQueryId()
            truth = qpi.truths[index]
            total = images.count(truth)
            cutMinimum = 1
            cut = cutMaxIncrease + cutMinimum
            resultMax = proc.runNext(smoothing, cut)
            print("Quantity of images with ID = " + truth + " to find: " + str(total) + ", running query index: " + str(index))
            
            correctPerCutIncrease = [0] * (cutMaxIncrease + 2)
            precisionPerCutIncrease = [0] * (cutMaxIncrease + 2)
            for ind, ranking in enumerate(resultMax):
                imgid = images[int(ranking[0])]
                cutIdx = 0 if ind < cutMinimum else ind-cutMinimum
                correct = correctPerCutIncrease[cutIdx] # Get previous correct number of guesses and continue from that
                precision = precisionPerCutIncrease[cutIdx] # Get previous precision value and continue from that
                if imgid == truth:
                    correct +=1
                    precision += round(correct/(ind+1),10)
                correctPerCutIncrease[cutIdx] = correct # Store result of this cut index
                precisionPerCutIncrease[cutIdx] = precision # Store result of this cut index
                correctPerCutIncrease[cutIdx+1] = correct # Let next iteration build upon this one
                precisionPerCutIncrease[cutIdx+1] = precision # Let next iteration build upon this one
                    
            for cutIdx in range(cutMaxIncrease+1):
                correct = correctPerCutIncrease[cutIdx]
                recall = round(correct/total, 10)
                precision = precisionPerCutIncrease[cutIdx]
                meanAveragePrecision = round(precision/(cutMinimum+cutIdx),10)
                precrec = totalPrecisionAndRecallPerCutIncrease[cutIdx]
                precrec[0]+=meanAveragePrecision
                precrec[1]+=recall
        
        # Write results
        filename = './imageresults/run_%d.txt' % smoothing
        with open(filename, 'w') as f:
            for cutIdx in range(cutMaxIncrease+1):
                precrec = totalPrecisionAndRecallPerCutIncrease[cutIdx]
                totalprec = precrec[0]
                totalrec = precrec[1]
                #filename = './imageresults/results_%d.txt' % smoothing
                #with open(filename, 'a') as f:
                #    qryIdx = 0
                #    for prec, rec in precrec:
                #        f.write("queryIndex=" + str(qryIdx) + ";cut=" + str(cutIdx) + ";precision=" + str(prec) + ";recall=" + str(rec) + "\n")
                #        totalprec += prec
                #        totalrec += rec
                #f.write("cut=" + str(cutIdx) + ";avg precision=" + str(totalprec/len(queries)) + ";avg recall=" + str(totalrec/len(queries)) + "\n")
                f.write(str(cutIdx) + "\t" + str(totalprec/len(queries)) + "\t" + str(totalrec/len(queries)) + "\n")    
    end = time.time()
    print(end - start)


def make_dir():
    if not os.path.exists('./imageresults'):
        os.makedirs('./imageresults')


if __name__ == '__main__':
    make_dir()
    main()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
