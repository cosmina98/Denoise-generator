import numpy as np
import networkx as nx
import os
import requests
import logging
from rdkit import Chem
from coco_grape.data_graphicalizer.mol.molecular_graphicalizer import SmilesMolecularGraphicalizer
import pandas as pd


logger = logging.getLogger(__name__)

def rdkmol_to_nx(mol):
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(), label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       label=str(int(b.GetBondTypeAsDouble())))
    return graph

def sdf_to_nx(file):
    suppl = Chem.SDMolSupplier(file)
    for mol in suppl:
        if mol:
            yield rdkmol_to_nx(mol)

def smi_to_nx(file):
    suppl = Chem.SmilesMolSupplier(file)
    for mol in suppl:
        yield rdkmol_to_nx(mol)

class RDKitMolFileLoader(object):
    def __init__(self, dirname='.', filetype='smi'):
        self.dirname = dirname
        self.filetype = filetype

    def load(self, filename):
        full_fname = os.path.join(self.dirname, filename)
        if self.filetype == 'sdf': graphs = list(sdf_to_nx(full_fname))
        elif self.filetype == 'smi': graphs = list(smi_to_nx(full_fname))
        return graphs        

class PubChemLoader(object):
    def __init__(self):
        self.root_uri = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
        self.pubchem_dir = 'PUBCHEM'

    def get_assay_description(self, assay_id):
        """get_assay_description."""
        fname = 'AID%s_info.txt' % assay_id
        full_fname = os.path.join(self.pubchem_dir, fname)
        if not os.path.isfile(full_fname):
            query = self.root_uri
            query += 'assay/aid/%s/summary/JSON' % assay_id
            reply = requests.get(query)
            text = reply.json()['AssaySummaries']['AssaySummary'][0]['Name']
            with open(full_fname, 'w') as file_handle:
                file_handle.write(text)
        else:
            with open(full_fname, 'r') as file_handle:
                text = ''
                for line in file_handle:
                    text += line
        return text

    def _get_compounds(self, fname, active, aid, stepsize=50):
        with open(fname, 'w') as file_handle:
            index_start = 0
            reply = requests.get(self._make_rest_query(aid, active=active))
            listkey = reply.json()['IdentifierList']['ListKey']
            size = reply.json()['IdentifierList']['Size']
            for chunk, index_end in enumerate(range(0, size + stepsize, stepsize)):
                if index_end != 0:
                    repeat = True
                    while repeat:
                        t = 'Chunk %s) Processing compounds %s to %s (%s)' % (chunk, index_start, index_end - 1, size)
                        logger.debug(t)
                        query = self.root_uri
                        query += 'compound/listkey/' + str(listkey)
                        query += '/SDF?&listkey_start=' + str(index_start)
                        query += '&listkey_count=' + str(stepsize)
                        reply = requests.get(query)
                        if 'PUGREST.Timeout' in reply.text: logger.debug("PUGREST TIMEOUT")
                        elif "PUGREST.BadRequest" in reply.text:
                            # bad request means that the server throw our
                            # molecule list away.
                            # we just make a new one
                            logger.debug('bad request %s %d %d %d' % (query, chunk, index_end, size))
                            reply = requests.get(self._make_rest_query(aid, active=active))
                            listkey = reply.json()['IdentifierList']['ListKey']
                        elif reply.status_code != 200:
                            logger.debug("UNKNOWN ERRA " + query)
                            logger.debug(reply.status_code)
                            logger.debug(reply.text)
                            raise Exception('UNKNOWN')
                        else:  # everything is OK
                            repeat = False
                            file_handle.write(reply.text)
                index_start = index_end
            logger.debug('compounds available in file: ', fname)

    def _make_rest_query(self, assay_id, active=True):
        if active: mode = 'active'
        else: mode = 'inactive'
        core = 'assay/aid/%s/cids/JSON?cids_type=%s&list_return=listkey' % (assay_id, mode)
        rest_query = self.root_uri + core
        return rest_query

    def _query_db(self, assay_id, fname=None, active=True, stepsize=50):
        self._get_compounds(fname=fname + ".tmp", active=active, aid=assay_id, stepsize=stepsize)
        os.rename(fname + '.tmp', fname)

    def download(self, assay_id, active=True, stepsize=50):
        """download."""
        if not os.path.exists(self.pubchem_dir): os.mkdir(self.pubchem_dir)
        if active: fname = 'AID%s_active.sdf' % assay_id
        else: fname = 'AID%s_inactive.sdf' % assay_id
        full_fname = os.path.join(self.pubchem_dir, fname)
        if not os.path.isfile(full_fname):
            logger.debug('Querying PubChem for AID: %s' % assay_id)
            self._query_db(assay_id, fname=full_fname, active=active, stepsize=stepsize)
        else: logger.debug('Reading from file: %s' % full_fname)
        return full_fname

    def load(self, assay_id, dirname='PUBCHEM', format_type='sdf'):
        self.download(assay_id, active=True, stepsize=50)
        self.download(assay_id, active=False, stepsize=50)
        fname_active = 'AID%s_active.sdf' % assay_id
        fname_inactive = 'AID%s_inactive.sdf' % assay_id
        if format_type == 'sdf':
            pos_graphs = RDKitMolFileLoader(dirname=dirname, filetype='sdf').load(fname_active)
            neg_graphs = RDKitMolFileLoader(dirname=dirname, filetype='sdf').load(fname_inactive)
        elif format_type == 'smi':
            pos_graphs = RDKitMolFileLoader(dirname=dirname, filetype='smi').load(fname_active)
            neg_graphs = RDKitMolFileLoader(dirname=dirname, filetype='smi').load(fname_inactive)
        graphs = pos_graphs+neg_graphs
        targets = [1]*len(pos_graphs)+[0]*len(neg_graphs)
        return graphs, targets


def load_pubchem(assay_id):
    #assay_ids = ['2631','624249','651741','588350','463230','492952','743219','492992','463213']
    graphs, targets = PubChemLoader().load(assay_id)
    return graphs, targets

def load_pubchem_492992(): return load_pubchem('492992')


class CSVSmileFileLoader(object):
    def __init__(self, file_name='', smile_column='smile', target_column=None):
        self.file_name = file_name
        self.smile_column = smile_column
        self.target_column = target_column

    def load(self, size=None):
        df = pd.read_csv(self.file_name)
        targets = df[self.target_column].values
        smis = df[self.smile_column].values
        if size is not None:
            smis = smis[:size]
            targets = targets[:size]
        graphs, targets = SmilesMolecularGraphicalizer().fit_transform(smis, targets)
        return graphs, targets 
    
def load_csv_data(): return CSVSmileFileLoader(file_name='cancer_v2.csv', smile_column='smile', target_column='AVERAGE_GI50').load(size=1000)

