#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import scipy as sp
import networkx as nx

from eden.graph import vectorize as eden_vectorize
from eden.sequence import vectorize as eden_sequence_vectorize

from toolz import partition_all
import multiprocessing_on_dill as mp

class EdenSequenceVectorizer(object):
    def __init__(self,
                 radius=1,
                 distance=6,
                 nbits=14,
                 parallel=True):
        self.radius = radius
        self.distance = distance
        self.nbits = nbits
        self.parallel = parallel
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, seqs, targets=None):
        return self
    
    def transform_parallel(self, seqs):
        n_cpus = mp.cpu_count()
        batch_size = len(seqs)//n_cpus
        if len(seqs) < n_cpus: seqs_list = [seqs]
        else: seqs_list = list(partition_all(batch_size, seqs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, seqs_list)
        pool.close()
        data_mtx = sp.sparse.vstack(results)
        return data_mtx

    def transform_sequential(self, seqs):
        return eden_sequence_vectorize(seqs, r=self.radius, d=self.distance, nbits=self.nbits)

    def transform(self, seqs):
        if self.parallel:
            return self.transform_parallel(seqs)
        else:
            return self.transform_sequential(seqs)

    def fit_transform(self, seqs, targets=None):
        return self.fit(seqs, targets).transform(seqs)


class EdenGraphVectorizer(object):
    def __init__(self,
                 radius=1,
                 distance=6,
                 nbits=16,
                 use_attributes=False,
                 parallel=True):
        self.radius = radius
        self.distance = distance
        self.nbits = nbits
        self.use_attributes = use_attributes
        self.parallel = parallel
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        if self.parallel: return self.transform_parallel(graphs)
        else: return self.transform_sequential(graphs)

    def transform_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        batch_size = len(graphs)//n_cpus
        if batch_size < 2: graphs_list = [graphs]
        else: graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, graphs_list)
        pool.close()
        data_mtx = sp.sparse.vstack(results)
        return data_mtx

    def transform_sequential(self, graphs):
        if self.use_attributes is True: return eden_vectorize(graphs, r=self.radius, d=self.distance, nbits=self.nbits, discrete=True, key_vec='vec')
        return eden_vectorize(graphs, r=self.radius, d=self.distance, nbits=self.nbits)

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)
    
