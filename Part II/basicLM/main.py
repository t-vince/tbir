from parse import *
from query import *
import operator
import os
import subprocess
import sys
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    qpi = QueryParser(filename='./testqueries_val.txt')
    cpi = CorpusParser(filename='./target_collection_parsed.txt')
    qp = QueryParser(filename='./queries.txt')
    cp = CorpusParser(filename='./corpus.txt')
    #cp.parse()
    print('parsing corpus')
    cpi.readparsed()
    #qp.parse()
    print('parsing queries')
    qpi.parseImages()
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
    lines = OrderedDict()
    for index, result in enumerate(results):
        for mu, l in result.items():
            s = sorted([(k, v) for k, v in l.items()], key=operator.itemgetter(1))
            s.reverse()
            for rank, x in enumerate(s[:50]):
                tmp = index+1, x[0], rank+1, x[1]
                line = '{:<} Q0 {:<} {:<} {:<} NH-QL\n'.format(*tmp)
                if mu in lines:
                    lines[mu].append(line)
                else:
                    lines[mu] = [line]
        for mu, txt in lines.items():
            filename = './imageresults/run.%d' % mu
            with open(filename, 'w') as f:
                f.writelines(txt)


def make_dir():
    if not os.path.exists('./imageresults'):
        os.makedirs('./imageresults')


if __name__ == '__main__':
    make_dir()
    main()
    # subprocess.call(['java', '-jar', '../tool/eval.jar', '-d', '../results'])
