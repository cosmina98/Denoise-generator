import networkx as nx 
import numpy as np
import requests
import subprocess as sub
import random
from toolz import partition_all
import multiprocessing_on_dill as mp
from sklearn.neighbors import NearestNeighbors
import uuid
import os
import logging
from coco_grape.data_graphicalizer.rna import lib_forgi

logger = logging.getLogger('RNA')
logger.setLevel(logging.INFO)


def write_fasta(seqs, fname):
    with open(fname,'w') as f:
        for seq in seqs:
            h, s = seq
            f.write('>%s\n'%h)
            f.write('%s\n'%s)

def read(uri):
    """Abstract read function.

    It can accept a URL, a file path and a python list.
    In all cases an iterable object should be returned.
    """
    if isinstance(uri, list):
        # test if it is iterable: works for lists and generators, but not for
        # strings
        return uri
    else:
        try:
            # try if it is a URL and if we can open it
            f = requests.get(uri).text.split('\n')
        except ValueError:
            # assume it is a file object
            f = open(uri)
        return f


def is_iterable(test):
    """is_iterable."""
    if hasattr(test, '__iter__'):
        return True
    else:
        return False
    
def fasta_to_fasta(source, normalize=True):
    """Take a FASTA file and yield a normalised FASTA file.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    normalize : bool
        If True all characters are uppercased and Ts are replaced by Us
    """
    iterable = _fasta_to_fasta(source)
    for line in iterable:
        header = line
        seq = next(iterable)
        if normalize:
            seq = seq.upper()
            seq = seq.replace('T', 'U')
        yield header, seq


def _fasta_to_fasta(source):
    seq = ""
    for line in read(source):
        if line:
            if line[0] == '>':
                line = line[1:]
                if seq:
                    yield seq
                    seq = ""
                line_str = str(line)
                yield line_str.strip()
            else:
                line_str = line.split()
                if line_str:
                    seq += str(line_str[0]).strip()
    if seq:
        yield seq


def load_fasta(source, normalize=True):
    """Load sequences."""
    return fasta_to_fasta(source, normalize)


def generate_random_sequences(length=30, n_seqs=20, alphabet='ACGT'):
    return [''.join(random.choices(alphabet, k=length)) for i in range(n_seqs)]


def sequence_dotbracket_to_graph(seq_info=None, seq_struct=None):
    """Given a sequence and the dotbracket sequence make a graph.

    Parameters
    ----------
    seq_info string
        node labels eg a sequence string
    seq_struct  string
        dotbracket string

    Returns
    -------
        returns a nx.Graph
        secondary struct associated with seq_struct
    """
    graph = nx.Graph()
    lifo = list()
    for i, (c, b) in enumerate(zip(seq_info, seq_struct)):
        graph.add_node(i, label=c, position=i)
        if i > 0:
            graph.add_edge(i, i - 1, label='-', type='backbone', len=1)
        if b == '(':
            lifo.append(i)
        if b == ')':
            j = lifo.pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)
    return graph

def dotbracket_to_stem_loop_code(struct):
    bg = lib_forgi.BulgeGraph()
    bg.from_dotbracket(struct, None)
    stem_loop_code = bg.to_element_string()
    return stem_loop_code

def seq_to_graph(header, sequence, **options):
    """Fold a sequence in a path graph."""
    seq_struct = '.' * len(sequence)
    graph = sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=seq_struct)
    graph.graph['info'] = 'sequence'
    graph.graph['sequence'] = sequence
    graph.graph['structure'] = seq_struct
    graph.graph['secondary_structure'] = dotbracket_to_stem_loop_code(seq_struct)
    graph.graph['id'] = header
    label_nodes_with_secondary_structure(graph)
    if options.get('mode', 'standard') == 'non_stem': relabel_graph_to_non_stem(graph)
    return graph


def rnafold_wrapper(sequence, **options):
    """Wrap RNAfold."""
    # defaults
    flags = options.get('flags', '--noPS')
    # command line
    cmd = 'echo "%s" | RNAfold %s' % (sequence, flags)
    out = sub.check_output(cmd, shell=True)
    out = out.decode()
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct = text[1].split()[0]
    return seq_info, seq_struct

def seq_struct_to_networkx(header, sequence, dotbracket, **options):
    graph = sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=dotbracket)
    graph.graph['info'] = 'RNAfold'
    graph.graph['sequence'] = sequence
    graph.graph['structure'] = dotbracket
    graph.graph['secondary_structure'] = dotbracket_to_stem_loop_code(dotbracket)
    graph.graph['id'] = header
    label_nodes_with_secondary_structure(graph)
    if options.get('mode', 'standard') == 'non_stem': relabel_graph_to_non_stem(graph)            
    return graph

def _string_to_networkx(header, sequence, **options):
    seq_info, seq_struct = rnafold_wrapper(sequence, **options)
    graph = seq_struct_to_networkx(header, sequence, seq_struct, **options)
    return graph

def label_nodes_with_secondary_structure(graph):
    for u,s in zip(graph.nodes(), graph.graph['secondary_structure']):
        graph.nodes[u]['secondary_structure'] = s
        if s=='s': graph.nodes[u]['non_stem'] = 'S'
        else: graph.nodes[u]['non_stem'] = graph.nodes[u]['label']

def relabel_graph_to_non_stem(graph):
    for u,s in zip(graph.nodes(), graph.graph['secondary_structure']):
        if s=='s': graph.nodes[u]['label'] = 'S'
            
def rnafold_to_graph(iterable=None, **options):
    """Fold RNA seq with RNAfold.

    Parameters
    ----------
    iterable: over (header_string, sequence_string)

    options

    Returns
    -------
        nx.graph generator
    """
    graphs = []
    for header, seq in iterable:
        try:
            graph = _string_to_networkx(header, seq, **options)
        except Exception as e:
            logger.debug('%s' % e)
            logger.debug('Error in: %s' % str(seq))
            graph = seq_to_graph(header, seq)
        graphs.append(graph)
    return graphs

def fold(seqs):
    return rnafold_to_graph(seqs)

def sequence(seqs):
    return [seq_to_graph(header, sequence) for header, sequence in seqs]

def seqs_to_rnaseqs(seqs):
    rna_seqs = [('>%d'%i,seq) for i, seq in enumerate(seqs)]
    return rna_seqs

def seqs_to_graphs(seqs, mode='standard'):
    rnaseqs = seqs_to_rnaseqs(seqs)
    return [seq_to_graph(header, sequence, mode=mode) for header, sequence in rnaseqs]
    
def seqs_to_fold_graphs(seqs, mode='standard'):
    rnaseqs = seqs_to_rnaseqs(seqs)
    return rnafold_to_graph(rnaseqs, mode=mode)


class FastaSequenceGraphicalizer(object):
    def __init__(self):
        pass
    
    def fit(self, seqs, targets=None):
        return self
    
    def transform(self, seqs):
        return [seq_to_graph(header, sequence) for header, sequence in seqs]

    def fit_transform(self, seqs, targets=None):
        return self.fit(seqs, targets).transform(seqs)


class FastaRNAFoldGraphicalizer(object):
    def __init__(self, parallel=True):
        self.parallel = parallel
    
    def fit(self, seqs, targets=None):
        return self

    def transform(self, seqs):
        if self.parallel:
            return self.transform_parallel(seqs)
        else:
            return self.transform_sequential(seqs)

    def transform_parallel(self, seqs):
        n_cpus = mp.cpu_count()
        batch_size = len(seqs)//n_cpus
        if batch_size < 2:
            seqs_list = [seqs]
        else:    
            seqs_list = list(partition_all(batch_size, seqs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, seqs_list)
        pool.close()
        all_list_of_results = []
        for list_of_results in results:
            all_list_of_results.extend(list_of_results)
        return all_list_of_results

    def transform_sequential(self, seqs):
        return rnafold_to_graph(seqs)
    
    def fit_transform(self, seqs, targets=None):
        return self.fit(seqs, targets).transform(seqs)


class FastaRNAalignmentFoldGraphicalizer(object):
    def __init__(self, vectorizer=None, n_neighbors=5, parallel=True):
        self.vectorizer = vectorizer
        self.knn = NearestNeighbors(n_neighbors=n_neighbors+1)
        self.parallel = parallel
    
    def fit(self, seqs, targets=None):
        self.train_seqs = seqs
        sequences = [seq for header,seq in seqs]
        encodings = self.vectorizer.transform(sequences)
        self.knn.fit(encodings)
        return self

    def transform(self, seqs):
        if self.parallel:
            graphs = self.transform_parallel(seqs)
        else:
            graphs = self.transform_sequential(seqs)
        if os.path.exists('alirna.ps'): os.unlink('alirna.ps')
        return graphs

    def fit_transform(self, seqs, targets=None):
        graphs = self.fit(seqs, targets).transform(seqs)
        return graphs

    def transform_parallel(self, seqs):
        n_cpus = mp.cpu_count()
        batch_size = len(seqs)//n_cpus
        if batch_size < 2:
            seqs_list = [seqs]
        else:    
            seqs_list = list(partition_all(batch_size, seqs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, seqs_list)
        pool.close()
        all_list_of_results = []
        for list_of_results in results:
            all_list_of_results.extend(list_of_results)
        return all_list_of_results

    def transform_sequential(self, seqs):
        graphs = [self.transform_sequential_sequence(seq) for seq in seqs]
        return graphs 
    
    def transform_sequential_sequence(self, seq):
        unique_filename = str(uuid.uuid4())
        unique_filename_fasta = str(uuid.uuid4())
        fname = '%s.fasta'%unique_filename_fasta
        try:
            header, sequence = seq
            encodings = self.vectorizer.transform([sequence])
            distance_mtx, index_mtx = self.knn.kneighbors(encodings)
            neighs = index_mtx[:,1:][0]
            to_align_seqs = [seq]+[self.train_seqs[j] for j in neighs]
            write_fasta(to_align_seqs, fname)
            cmd = 'muscle -quiet -align %s.fasta -output %s.afa 2>/dev/null && RNAalifold %s.afa 2>/dev/null'%(unique_filename_fasta, unique_filename,unique_filename)
            out = sub.check_output(cmd, shell=True)
            out = out.decode()
            text = out.strip().split('\n')
            dotbracket = text[1].split()[0]
            aseqs = list(load_fasta('%s.afa'%unique_filename))
            aseq = aseqs[0][1]
            compact_aseq = ''.join([c for c in aseq if c not in '-'])
            compact_dotbracket = ''.join([d for c,d in zip(aseq, dotbracket) if c not in '-'])
            header = aseqs[0][0]
            sequence = compact_aseq
            dotbracket = compact_dotbracket
    
            graph = seq_struct_to_networkx(header, sequence, dotbracket)
        except Exception as e:
            logger.debug('%s' % e)
            logger.debug('Error in: %s' % str(seq))
            graph = rnafold_to_graph([seq])[0]
        if os.path.exists('%s.fasta'%unique_filename_fasta): os.unlink('%s.fasta'%unique_filename_fasta)
        if os.path.exists('%s.afa'%unique_filename): os.unlink('%s.afa'%unique_filename)
        return graph