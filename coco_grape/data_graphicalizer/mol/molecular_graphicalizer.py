from rdkit import Chem

import networkx as nx

def sdf_to_nx(file):
    # read sdf file
    suppl = Chem.SDMolSupplier(file)
    for mol in suppl:
        if mol:
            yield rdkmol_to_nx(mol)


def smi_to_nx(file, targets=None):
    if targets is None:
        for mol in Chem.SmilesMolSupplier(file):
            try:
                yield rdkmol_to_nx(mol)
            except:
                pass
    else:
        for mol,target in zip(Chem.SmilesMolSupplier(file), targets):
            try:
                yield rdkmol_to_nx(mol), target
            except:
                pass 

def rdkmol_to_nx(mol):
    #  rdkit-mol object to nx.graph
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(), label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       label=str(int(b.GetBondTypeAsDouble())))
    return graph


def smiles_strings_to_nx(smileslist, targets=None):
    # smiles strings
    if targets is None:
        effective_targets = [1]*len(smileslist)
    else:
        effective_targets = targets
    for smile, target in zip(smileslist, effective_targets):
        try:
            mol = Chem.MolFromSmiles(smile)
            if targets is None: yield rdkmol_to_nx(mol)
            else: yield rdkmol_to_nx(mol), target
        except:
            pass 


################
# exporting networkx graphs
###############

def nx_to_smi(graphs, file=None):
    # writes smiles strings to a file
    chem = [nx_to_rdkit(graph) for graph in graphs]
    smis = [Chem.MolToSmiles(m) for m in chem]
    if file:
        with open(file, 'w') as f:
            f.write('\n'.join(smis))
    return smis


def nx_to_sdf(graphs, file=None):
    # writes smiles strings to a file
    writer = Chem.rdmolfiles.SDWriter(file)
    list_of_mols = [nx_to_rdkit(graph) for graph in graphs]
    for mol in list_of_mols:
        writer.write(mol)
    return list_of_mols


def nx_to_inchi(graphs, file=None):
    # writes smiles strings to a file
    chem = [nx_to_rdkit(graph) for graph in graphs]
    inchis = [Chem.inchi.MolToInchi(m) for m in chem]
    if file:
        with open(file, 'w') as f:
            f.write('\n'.join(inchis))
    return inchis


def nx_to_rdkit(graph):
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        atom_index[n] = mw.AddAtom(Chem.Atom(d['label']))
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b]
        bond_type = d.get("label", '1')
        if bond_type == '1':
            mw.AddBond(start, end, Chem.BondType.SINGLE)
        elif bond_type == '2':
            mw.AddBond(start, end, Chem.BondType.DOUBLE)
        elif bond_type == '3':
            mw.AddBond(start, end, Chem.BondType.TRIPLE)
        # more options:
        # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdchem.BondType-class.html
        else:
            raise Exception('bond type not implemented')

    mol = mw.GetMol()
    return mol



class SmilesMolecularGraphicalizer(object):
    def __init__(self, file_name=None, targets=None, return_targets=False):
        self.return_targets = return_targets
        if file_name is not None: return self.read(file_name, targets)
    
    def fit(self, seqs, targets=None):
        return self

    def read(self, file_name, targets=None):
        if targets is None:
            graphs = list(smi_to_nx(file_name))
            return graphs
        else:
            graphs, sel_targets = zip(*[(graph, target) for graph, target in smi_to_nx(file_name, targets)])
            return graphs, sel_targets

    def transform(self, seqs, targets=None):
        if self.return_targets is False:
            graphs = list(smiles_strings_to_nx(seqs))
            return graphs
        else:
            graphs, sel_targets = zip(*[(graph, target) for graph, target in smiles_strings_to_nx(seqs, targets)])
            return graphs, sel_targets

    def fit_transform(self, seqs, targets=None):
        return self.fit(seqs, targets).transform(seqs, targets)

    def inverse_transform(self, graphs):
        chem = [nx_to_rdkit(graph) for graph in graphs]
        seqs = [Chem.MolToSmiles(m) for m in chem]
        return seqs
