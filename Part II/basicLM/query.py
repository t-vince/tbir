from rank import *
from collections import OrderedDict
import operator
import time



class QueryProcessor:
    def __init__(self, queries, idx, dlt, ft, score_function='BM25'):
        self.queries = queries
        #self.index, self.ft, self.dlt = build_data_structures(corpus)
        self.idx_file = idx
        self.dlt = dlt
        self.ft = ft
        self.score_function = score_function
        self.qid = 0;
        self.idx = {}
        with open(self.idx_file, buffering=1000000000) as idx:
            for line in idx:
                tmp = line.split()
                word = tmp.pop(0)
                freq = [tuple(x.split(':')) for x in tmp]
                self.idx[word] = freq
        
    def hasNext(self):
        return self.qid < len(self.queries)
        
    def runNext(self, smoothing, cut):
        if self.score_function == 'BM25':
            result = self.run_BM25(query)
        elif self.score_function == 'Query Likelihood':
            query = self.queries[self.qid]
            self.qid += 1
            newResults = self.run_query_likelihood(query, smoothing)            
            s = sorted([(k, v) for k, v in newResults.items()], key=operator.itemgetter(1))
            s.reverse()
            result = s[:cut]
        return result
        
    def restart(self):
        self.qid = 0
        
    def getCurrentQueryId(self):
        return self.qid

    def run_query_likelihood(self, query, smoothing):
        #print('matching query: %d %s'% (self.qid, query))
        results = OrderedDict() # collect document rankings for this value of mu
        for term in query:
            if not term in self.idx:
                continue
            
            freq = self.idx[term]
            termFreq = self.ft.get_frequency(term)
            score_query_likelihood = QueryLikelihood(mu=smoothing, c=termFreq, C=len(self.ft), D=len(self.dlt))
            docs = set()
            
            for docid, f in freq:
               docs.add(docid)
               #score = score_query_likelihood(f=float(f), mu=smoothing, c=termFreq, C=len(self.ft), D=len(self.dlt))
               score = score_query_likelihood.get_score(f)               
               if docid in results:
                   results[docid] += score
               else:
                   results[docid] = score
            #score documents that don't contain term
            tmp = [str(x) for x in range(len(self.dlt))]
            s = set(tmp).difference(docs)
            #score = score_query_likelihood(f=0, mu=smoothing, c=self.ft.get_frequency(term), C=len(self.ft), D=len(self.dlt))
            
            smoothingscore = score_query_likelihood.get_score(0)                        
            for docid in s:
                if docid in results:
                    results[docid] += smoothingscore
                else:
                    results[docid] = smoothingscore
        return results
