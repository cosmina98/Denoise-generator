import pandas as pd
import networkx as nx
import numpy as np
from rdkit import Chem
from ogb.graphproppred import Evaluator
from ogb.graphproppred import GraphPropPredDataset
from ogb.lsc import PCQM4Mv2Dataset

def rdkmol_to_nx(mol):
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(), label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), label=str(int(b.GetBondTypeAsDouble())))
    return graph

def smiles_strings_to_nx(smileslist):
    for smile in smileslist:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            yield rdkmol_to_nx(mol)
        else:
            yield rdkmol_to_nx(Chem.MolFromSmiles('C'))
            
def load_ogbg_PCQM4M(num_mols=None):
    dataset = PCQM4Mv2Dataset(root = 'mol_dataset', only_smiles = True)
    if num_mols is None: num_mols = len(dataset)
    mols, targets = zip(*[dataset[i] for i in range(num_mols)])
    graphs = list(smiles_strings_to_nx(mols))
    targets = np.array(targets).reshape(-1,1)
    return graphs, targets

def ogb2networkx(graph, use_first_attribute_as_label=True):
    G = nx.Graph()
    n = graph['num_nodes']
    G.add_nodes_from(range(n))
    node_attributes = graph['node_feat']
    if node_attributes is not None:    
        for u,att in zip(G.nodes(),node_attributes):
            if use_first_attribute_as_label is True:
                G.nodes[u]['label']=att[0]
            else:
                G.nodes[u]['label']=1
            G.nodes[u]['vec']=att
    else:
        for u in G.nodes():
            G.nodes[u]['label']=1
            G.nodes[u]['vec']=np.array([1])
    src, dst = graph['edge_index'][0], graph['edge_index'][1]
    edge_attributes = graph['edge_feat']
    if edge_attributes is not None:
        for u,v,att in zip(src, dst, edge_attributes):
            if use_first_attribute_as_label is True:
                G.add_edge(u,v, label=att[0]+1, vec=att)
            else:
                G.add_edge(u,v, label=1, vec=att)
    else:
        for u,v in zip(src, dst):
            G.add_edge(u,v, label=1, vec=[1])
    return G

def ogb2graphs(ogb_graphs, use_first_attribute_as_label=True):
    graphs = [ogb2networkx(ogb_graph, use_first_attribute_as_label) for ogb_graph in ogb_graphs]
    return graphs 

def load_obg(name, target_id=0, use_int_target=True):
    names = ['ogbg-code2','ogbg-ppa','ogbg-molpcba','ogbg-molhiv','ogbg-moltox21','ogbg-molbace','ogbg-molbbbp','ogbg-molclintox','ogbg-molmuv','ogbg-molsider','ogbg-moltoxcast']
    dataset = GraphPropPredDataset(name=name)
    #fill missing values
    targets = pd.DataFrame(dataset.labels).fillna(0).values
    graphs = ogb2graphs(dataset.graphs, use_first_attribute_as_label=True)
    targets = targets[:,target_id]
    if use_int_target: targets = targets.astype(int)
    return graphs, targets 

def load_ogbg_molbace(): return load_obg('ogbg-molbace', target_id=0, use_int_target=True)
