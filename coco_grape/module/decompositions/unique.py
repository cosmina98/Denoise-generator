#!/usr/bin/env python
"""Provides interface."""

import networkx as nx 
import numpy as np
from toolz import curry
from coco_grape.module.graph_hash import graph_hash
from coco_grape.module.construct import make_multi_decomposition_function

@curry
@make_multi_decomposition_function
def unique_decomposition_function(subgraphs, basegraph=None, **args):
    components = []
    component_idxs = []
    
    if len(subgraphs) == 0:
        return components, component_idxs

    if len(subgraphs) == 1:
        component = [u for u in subgraphs[0].nodes()]
        components = [component]
        component_idxs = [[0]]
        return components, component_idxs

    
    #keep only one copy of each subgraph
    hashes = [hash(tuple(sorted(subgraph.nodes()))) for subgraph in subgraphs]
    idxs = np.argsort(hashes)
    for i in range(1,len(idxs)):
        if hashes[idxs[i-1]] != hashes[idxs[i]]:
            component_idxs.append([idxs[i-1]])
            component = list(subgraphs[idxs[i-1]].nodes())
            components.append(component)
    if hashes[idxs[i]] != hashes[idxs[i-1]]:
        component_idxs.append([idxs[i]])
        component = list(subgraphs[idxs[i]].nodes())
        components.append(component)
    return components, component_idxs

unique = unique_decomposition_function()