import json
from rdkit import Chem
from collections import defaultdict
import numpy as np
import pickle
import tqdm

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

radius = 2
ngram = 3

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict



def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                # consider edge as bond
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            # # nodes = fingerprints

            # """Also update each edge ID considering two nodes
            # on its both sides."""
            # _i_jedge_dict = defaultdict(lambda: [])
            # for i, j_edge in i_jedge_dict.items():
            #     for j, edge in j_edge:
            #         both_side = tuple(sorted((nodes[i], nodes[j])))
            #         edge = edge_dict[(both_side, edge)]
            #         _i_jedge_dict[i].append((j, edge))
            # i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

with open('./data/Kcat_combination_0918.json', 'r') as infile :
    Kcat_data = json.load(infile)
    
compound_fingerprints = list()
adjacencies = list()

for data in tqdm.tqdm(Kcat_data) :
    smiles = data['Smiles']
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) # Add hydrogens
    atoms = create_atoms(mol) # Get atom features

    i_jbond_dict = create_ijbonddict(mol) # Get graph structure
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius) # Extract fingerprints

    compound_fingerprints.append(fingerprints)

    adjacency = create_adjacency(mol)
    adjacencies.append(adjacency)

np.save('./data/compound_fingerprints', compound_fingerprints)  
np.save('./data/adjacencies', adjacencies)

dump_dictionary(atom_dict, './data/atom_dict.pickle')
dump_dictionary(bond_dict, './data/bond_dict.pickle')
dump_dictionary(fingerprint_dict, './data/fingerprint_dict.pickle')
dump_dictionary(edge_dict, './data/edge_dict.pickle')

    




