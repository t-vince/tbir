__author__ = 'Nick Hirakawa'


from invdx import build_data_structures
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
        
        

    def run(self, smoothing, cut):
        results = []
        self.qid = 0
        
        for query in self.queries:
            start = time.time()
            if self.score_function == 'BM25':
                results.append(self.run_BM25(query))
            elif self.score_function == 'Query Likelihood':
                print('running query %d' % qid)
                qid += 1
                newResults = self.run_query_likelihood(query, smoothing)
                
                s = sorted([(k, v) for k, v in newResults.items()], key=operator.itemgetter(1))
                s.reverse()
                results.append(s[:cut])
            end = time.time()
            print(end - start)
        return results
    
    def hasNext(self):
        return self.qid < len(self.queries)
        
    def runNext(self, smoothing, cut):
        start = time.time()
        if self.score_function == 'BM25':
            result = self.run_BM25(query)
        elif self.score_function == 'Query Likelihood':
            print('running query %d' % self.qid)
            query = self.queries[self.qid]
            self.qid += 1
            newResults = self.run_query_likelihood(query, smoothing)
            
            s = sorted([(k, v) for k, v in newResults.items()], key=operator.itemgetter(1))
            s.reverse()
            result = s[:cut]
        end = time.time()
        print(end - start)
        return result
        
    def restart(self):
        self.qid = 0
        
    def getCurrentQueryId(self):
        return self.qid

    def run_query_likelihood(self, query, smoothing):
        print('query:', query)
        mu_result = OrderedDict() # collect document rankings for this value of mu
        for term in query:
            #print 'searching index'
            if not term in self.idx:
                continue
            
            freq = self.idx[term]
            termFreq = self.ft.get_frequency(term)
            score_query_likelihood = QueryLikelihood(mu=smoothing, c=termFreq, C=len(self.ft), D=len(self.dlt))
            
            print('parsing index word:', term)
            docs = set()
            #print 'scoring documents'
            #score documents that contain term
            for docid, f in freq:
               docs.add(docid)
               #score = score_query_likelihood(f=float(f), mu=smoothing, c=termFreq, C=len(self.ft), D=len(self.dlt))
               score = score_query_likelihood.get_score(f)
               
               if docid in mu_result:
                   mu_result[docid] += score
               else:
                   mu_result[docid] = score
            #print 'scoring other documents'
            #score documents that don't contain term
            tmp = [str(x) for x in range(len(self.dlt))]
            s = set(tmp).difference(docs)
            #score = score_query_likelihood(f=0, mu=smoothing, c=self.ft.get_frequency(term), C=len(self.ft), D=len(self.dlt))
            score = score_query_likelihood.get_score(f)
                                       
            for docid in s:
                if docid in mu_result:
                    mu_result[docid] += score
                else:
                    mu_result[docid] = score
            break
        return mu_result

'''
    def run_BM25(self, query):
        query_result = dict()
        for term in query:
            if term in self.index:
                doc_dict = self.index[term] # retrieve index entry
                for docid, freq in doc_dict.iteritems(): #for each document and its word frequency
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
                                       dl=self.dlt.get_length(docid),
                                       avdl=self.dlt.get_average_length()) # calculate score
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result

    def run_QueryLikelihood(self, query):
        query_result = OrderedDict()
        for mu in mu_values:
            mu_result = dict()
            for term in query:
                if term in self.index:
                    docs = set()
                    #score documents containing term
                    for docid, freq in self.index[term].iteritems():
                        score = score_query_likelihood(f=freq, mu=mu, c=self.ft.get_frequency(term),
                                                       C=len(self.index), D=len(self.dlt))
                        docs.add(docid)
                        if docid in mu_result:
                            mu_result[docid] += score
                        else:
                            mu_result[docid] = score
                    a = [str(x) for x in range(len(self.dlt))]
                    s = set(a).difference(docs)
                    #score documents not containing term
                    for docid in s:
                        score = score_query_likelihood(f=0, mu=mu, c=self.ft.get_frequency(term), C=len(self.index), D=len(self.dlt))
                        if docid in mu_result:
                            mu_result[docid] += score
                        else:
                            mu_result[docid] = score
                else:
                    score = score_query_likelihood(f=0, mu=mu, c=self.ft.get_frequency(term), C=len(self.index), D=len(self.dlt))
                    for i in range(len(self.dlt)):
                        if i in mu_result:
                            mu_result[i] += score
                        else:
                            mu_result[i] = score
            query_result[mu] = mu_result
        return query_result
    '''

