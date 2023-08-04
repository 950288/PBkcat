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

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i in range(len(nodes)):
                neighbors = i_jedge_dict.get(i, []) 
                if not neighbors:
                    fingerprint = (nodes[i], ())  # empty tuple
                else:
                    fingerprint = (nodes[i], tuple([(nodes[j], bond) for j, bond in neighbors]))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in enumerate(i_jedge_dict.values()):
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict                 

    return np.array(fingerprints)

def create_adjacency(mol):
    """Create a adjacency matrix from a molecular graph."""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return adjacency

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def save_array(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

if __name__ == "__main__":

    with open('./data/Kcat_combination_0918.json', 'r') as infile :
        Kcat_data = json.load(infile)
        
    compound_fingerprints = []
    adjacencies = []
    Kcats = []

    """Get the fingerprints and adjacency matrix of each compound."""
    for data in tqdm.tqdm(Kcat_data) :
        smiles = data['Smiles']
        Kcats.append(float(data['Value']))
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) # Add hydrogens
        atoms = create_atoms(mol) # Get atom features

        i_jbond_dict = create_ijbonddict(mol) # Get graph structure
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius) # Extract fingerprints
        compound_fingerprints.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

    save_array(Kcats, './data/Kcats.pickle')
    save_array(compound_fingerprints, './data/compound_fingerprints.pickle')
    save_array(adjacencies, './data/adjacencies.pickle')
    dump_dictionary(atom_dict, './data/atom_dict.pickle')
    dump_dictionary(bond_dict, './data/bond_dict.pickle')
    dump_dictionary(fingerprint_dict, './data/fingerprint_dict.pickle')
    dump_dictionary(edge_dict, './data/edge_dict.pickle')

    print('compound_fingerprints, adjacencies, atom_dict, bond_dict, fingerprint_dict, edge_dict saved successfully!')
    




