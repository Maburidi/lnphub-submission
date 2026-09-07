from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_LIST = list(range(1, 119))
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def smiles_to_graph(smiles: str, y: float | None = None) -> Data | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    atoms = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atoms.append([
            ATOM_LIST.index(atomic_num) if atomic_num in ATOM_LIST else 0,
            min(atom.GetTotalDegree(), 5),
            min(abs(atom.GetFormalCharge()) + 5, 10),
            int(atom.GetIsAromatic()),
        ])

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        btype = BOND_LIST.index(bt) if bt in BOND_LIST else 0
        conj = int(bond.GetIsConjugated())
        ring = int(bond.IsInRing())
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [a, b]
        col += [b, a]
        edge_feat += [[btype, conj, ring], [btype, conj, ring]]

    data = Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=torch.tensor([row, col], dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.long).view(-1, 3),
    )
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


def build_graphs(smiles, y):
    graphs = []
    keep = []
    for i, (smi, yi) in enumerate(zip(smiles, y)):
        if not np.isfinite(yi):
            continue
        graph = smiles_to_graph(smi, yi)
        if graph is None:
            continue
        graphs.append(graph)
        keep.append(i)
    return graphs, np.asarray(keep, dtype=int)
