import pandas as pd
import networkx as nx
import numpy as np
from rdkit import Chem
from tdc.single_pred import Tox
from tdc.single_pred import HTS
from tdc.single_pred import ADME
from tdc.utils import retrieve_label_name_list

def rdkmol_to_nx(mol):
    #  rdkit-mol object to nx.graph
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(), label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       label=str(int(b.GetBondTypeAsDouble())))
    return graph

def smiles_strings_to_nx(smileslist):
    # smiles strings
    for smile in smileslist:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            yield rdkmol_to_nx(mol)
        else:
            yield rdkmol_to_nx(Chem.MolFromSmiles('C'))
            
def load_tdc(name, problem_type='adme'):
    # pip install PyTDC
    # Therapeutics Data Commons (Artificial intelligence foundation for therapeutic science)
    # https://tdcommons.ai/single_pred_tasks/overview/
    try:
        label_list = retrieve_label_name_list(name)
        if problem_type=='tox': data = Tox(name=name, label_name=label_list[0])
        if problem_type=='hts': data = HTS(name=name, label_name=label_list[0])    
        if problem_type=='adme': data = ADME(name=name, label_name=label_list[0])    
    except:
        if problem_type=='tox': data = Tox(name=name)
        if problem_type=='hts': data = HTS(name=name)
        if problem_type=='adme': data = ADME(name=name)  
    smis = list(data.get_data()['Drug'])
    graphs = list(smiles_strings_to_nx(smis))
    targets = list(data.y)
    graph_with_no_edges_ids = [i for i, g in enumerate(graphs) if nx.number_of_edges(g) == 0]
    graphs = [graph for i, graph in enumerate(graphs) if i not in graph_with_no_edges_ids]
    targets = [target for i, target in enumerate(targets) if i not in graph_with_no_edges_ids]
    return graphs, targets

adme_absorption_datasets = ['Caco2_Wang', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv']
adme_distribution_datasets = ['BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo']
adme_metabolism_datasets = ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP_1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels']
adme_execretion_datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']
tox_datasets = ['hERG', 'AMES', 'DILI', 'skin_reaction', 'LD50_Zhu', 'Carcinogens_Lagunin', 'ToxCast', 'Tox21', 'ClinTox']
hts_datasets = ['HIV', 'sarscov2_3clpro_diamond', 'sarscov2_vitro_touret', 'orexin1_receptor_butkiewicz', 'm1_muscarinic_receptor_agonists_butkiewicz', 'm1_muscarinic_receptor_antagonists_butkiewicz', 'potassium_ion_channel_kir2.1_butkiewicz', 'kcnq2_potassium_channel_butkiewicz', 'cav3_t-type_calcium_channels_butkiewicz', "choline_transporter_butkiewicz", 'serine_threonine_kinase_33_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz']

def load_tdc_adme_absorption_Lipophilicity_AstraZeneca(): return load_tdc('Lipophilicity_AstraZeneca', problem_type='adme')
def load_tdc_tox_AMES(): return load_tdc('AMES', problem_type='tox')
def load_tdc_hts_HIV(): return load_tdc('HIV', problem_type='hts')

