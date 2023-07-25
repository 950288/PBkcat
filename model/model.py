import torch
import torch.nn as nn
import pickle
import math
import numpy as np



class KcatPrediction(nn.Module):
    def __init__(self , args):
        super().__init__()
        dim = args['dim']
        layer_output = args['layer_output']
        layer_gnn = args['layer_gnn']
        layer_dnn = args['layer_dnn']
        len_fingerprint = args['len_fingerprint']
        self.embed_fingerprint = nn.Embedding(len_fingerprint, dim)
        self.W_gnn = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(layer_gnn)
        ])
        # self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5)
        # self.pooling = torch.nn.MaxPool2d(2)
        self.dnn = nn.Linear(512*8943, dim)
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

        """Protein vector with CNN."""
        protein_vector = protein_vector[:512]
        protein_vector = np.pad(protein_vector,((0,512-len(protein_vector)),(0,0)),'constant',constant_values = (0,0))
        # protein_vector = torch.from_numpy(protein_vector)
        # protein_vector = self.conv1(protein_vector)
        # protein_vector = self.pooling(protein_vector)
        # protein_vector = self.conv2(protein_vector)
        # protein_vector = self.pooling(protein_vector)
        protein_vector = torch.from_numpy(protein_vector)
        protein_flatten = torch.flatten(protein_vector) 
        # self.dnn = nn.Linear(512*8943, dim)
        protein_flatten = self.dnn(protein_flatten) 
        
        """Concatenate the two vector and output the interaction."""
        cat_vector = torch.cat((compound_vector, compound_flatten), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# class Tester(object):



if __name__ == '__main__':

    


    
    model = KcatPrediction().to(device)

    dataset = list(zip(compound_fingerprints, adjacencies, proteins))

    logits = model(dataset[0])

    print('%.4f' %math.pow(2,logits))


        


