import numpy as np
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as normalize_func
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
 
class TextVectorizer(object):
    def __init__(self, n_components=None, normalize=True, stop_words='english', analyzer='word', ngram_range=(1,1), use_tfidf=True, use_counts=False, use_hashing=False, n_features=2**16):
        self.n_components = n_components
        self.normalize = normalize
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.use_counts = use_counts
        self.use_hashing = use_hashing
        self.n_features = n_features
        args = dict(decode_error='ignore',stop_words=stop_words, analyzer=analyzer, ngram_range=ngram_range, strip_accents='ascii')
        if analyzer != 'word':
            stop_words = None
        if use_hashing:
            self.vectorizer = HashingVectorizer(n_features=n_features, **args)
        if use_counts:
            self.vectorizer = CountVectorizer(**args)
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(**args)
        if self.n_components is not None:
            self.svd = TruncatedSVD(n_components=self.n_components)
        else:
            self.svd = None

    def __repr__(self):
        infos=[]
        infos.append('n_components:%d'%self.n_components)
        if self.normalize:
            infos.append('normalize')
        infos.append('stop_words:%s'%self.stop_words)
        infos.append('analyzer:%s'%self.analyzer)
        infos.append('ngram_range:%s'%str(self.ngram_range))
        if self.use_tfidf:
            infos.append('use_tfidf')
        if self.use_counts:
            infos.append('use_counts')
        if self.use_hashing:
            infos.append('use_hashing')
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def _make_corpus(self, documents, targets):
        # concatenate all strings belonging to the same class in a single document
        corpus = []
        for reference_target in sorted(set(targets)):
            documents_target_i = [document for document, target in zip(documents, targets) if target == reference_target]
            corpus.append(' '.join(documents_target_i))
        return corpus

    def fit(self, documents, targets=None):
        if self.use_tfidf and targets is not None:
            corpus = self._make_corpus(documents, targets)
        else:
            corpus = documents 
        self.vectorizer = self.vectorizer.fit(corpus)
        if self.svd is not None:
            X = self.vectorizer.transform(documents)
            self.svd = self.svd.fit(X)
        return self
    
    def transform(self, documents):
        X = self.vectorizer.transform(documents)
        if self.svd is not None:
            X = self.svd.transform(X)
        if self.normalize:
            X = normalize_func(X)
        return X
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names() 
    


class OpenAITextEmbeddingVectorizer(object):
    def __init__(self, OPENAI_API_KEY=None, cache=None, max_num_words_per_document=1000):
        self.OPENAI_API_KEY = OPENAI_API_KEY
        if OPENAI_API_KEY is not None: self.llm = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        if cache is None: self.cache = dict()
        else: self.cache = cache
        self.max_num_words_per_document = max_num_words_per_document
        
    def fit(self, documets, targets=None):
        return self
    
    def transform_single(self, document):
        document_ = ' '.join(document.split()[:self.max_num_words_per_document])
        key = hash(document_)
        if key not in self.cache: self.cache[key] = self.llm.embed_documents([document_])[0]
        return self.cache[key]
    
    def transform(self, documents):
        return np.array([self.transform_single(document) for document in documents])
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)
    
    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self