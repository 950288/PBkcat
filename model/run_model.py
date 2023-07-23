import torch
import torch.nn as nn
import pickle
import numpy as np

class KcatPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(len_fingerprint, dim)
        self.W_gnn = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(layer_gnn)
        ])
        self.W_out = nn.ModuleList([
            nn.Linear(2*dim, 2*dim)                        
            for _ in range(layer_output)
        ])
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A.float(), hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def forward(self,inputs):

        fingerprints, adjacency, protein_vector = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Concatenate the two vector and output the interaction."""
        print(compound_vector.shape, protein_vector.shape)
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def load_tensor(file_name):
    return torch.from_numpy(np.load(file_name + '.npy', allow_pickle=True))

if __name__ == '__main__':

    dim = 10
    layer_output=3
    layer_gnn=3
    layer_cnn=3

    dir_input = './data/'
    compound_fingerprints = load_tensor(dir_input + 'compound_fingerprints')
    adjacencies = load_tensor(dir_input + 'adjacencies')
    proteins = load_tensor(dir_input + 'global_representations')

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    len_fingerprint = len(fingerprint_dict)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    model = KcatPrediction().to(device)

    dataset = list(zip(compound_fingerprints, adjacencies, proteins))

    logits = model(dataset[0])

    print(logits)


        


