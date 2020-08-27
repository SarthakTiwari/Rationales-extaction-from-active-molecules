import torch
from rdkit import Chem
import math
from torch import nn
from preprocessing import smile_to_graph,molgraph_collate_fn
from  edge_memory_network import EMNImplementation
m = nn.Sigmoid()


def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: 
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls

    
def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots

def extract_subgraph(smiles, selected_atoms): 
    
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
    subgraph = Chem.MolFromSmiles(subgraph)

    mol = Chem.MolFromSmiles(smiles)  
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    
    subgraph, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)
    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None

def scoring_function(smiles_list):
    scores=[]
    for smiles in smiles_list:
        adjacency, nodes, edges = smile_to_graph(smiles)
        adjacency, nodes, edges=molgraph_collate_fn(((adjacency, nodes, edges),))
        output=model.forward(nodes, edges,adjacency)
        scores.append(float(m(output)))
    return scores

class MCTSNode():

    def __init__(self, smiles, atoms, W=0, N=0, P=0):
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)

def mcts_rollout(node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function):
  
    cur_atoms = node.atoms
    if len(cur_atoms) <= MIN_ATOMS:
        return node.P

    
    if len(node.children) == 0:
        cur_cls = set( [i for i,x in enumerate(clusters) if x <= cur_atoms] )
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                
                if new_smiles in state_map:
                    new_node = state_map[new_smiles] 
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0: return node.P  

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score
        
    sum_count = sum([c.N for c in node.children])
    selected_node = max(node.children, key=lambda x : x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1
    return v

def mcts(smiles, scoring_function, n_rollout, max_atoms, prop_delta): 
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i,cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - set([i])
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    
    root = MCTSNode( smiles, set(range(mol.GetNumAtoms())) ) 
    state_map = {smiles : root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function)

    rationales = [node for _,node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]
    return smiles, rationales


if __name__=='__main__':
# pretrained model for jnk3
    model = EMNImplementation(node_features=40, edge_features=10,edge_embedding_size=50, message_passes=6, out_features=1,
            edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
    checkpoint = torch.load(r"checkpoints/jnk3.ckpt")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    MIN_ATOMS = 15
    C_PUCT = 10
    with open("jnk3_actives.txt") as f:
         for line in f:
             data = [line.split(',')[0] for line in f]

    result=[]
    for smiles in data:
        smile, rationales=mcts(smiles, scoring_function, n_rollout=20, max_atoms=20, prop_delta=0.5)
        result.append([smile,rationales])

    rset = set()
    for orig_smiles, rationales in result:
        rationales = sorted(rationales, key=lambda x:len(x.atoms))
#two rationales per molecule
        for x in rationales[:2]:
            if x.smiles not in rset:
               print(orig_smiles, x.smiles, len(x.atoms), x.P)
               rset.add(x.smiles)
    with open("rationales.txt", 'w') as f:
        for r in rset:
            f.write(str(r) + '\n')