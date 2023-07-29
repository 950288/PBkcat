import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error,r2_score
import tqdm

class KcatPrediction(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.dim = args['dim']
        self.layer_output = args['layer_output']
        self.layer_gnn = args['layer_gnn']
        self.layer_dnn = args['layer_dnn']
        self.lr = args['lr']
        self.len_fingerprint = args['len_fingerprint']
        self.weight_decay = args['weight_decay']
        self.embed_fingerprint = nn.Embedding(self.len_fingerprint, self.dim)
        self.W_gnn = nn.ModuleList([
            nn.Linear(self.dim, self.dim)
            for _ in range(self.layer_gnn)
        ])
        # self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5)
        # self.pooling = torch.nn.MaxPool2d(2)
        self.dnn = nn.Linear(8943, self.dim)
        self.W_out = nn.ModuleList([
            nn.Linear(2*self.dim, 2*self.dim)                        
            for _ in range(self.layer_output)
        ])
        self.W_interaction = nn.Linear(2*self.dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A.float(), hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def forward(self,inputs):

        fingerprints, adjacency, protein_flatten = inputs
        fingerprints = torch.LongTensor(fingerprints).to(self.device)
        adjacency = torch.FloatTensor(adjacency).to(self.device)
        protein_flatten = torch.FloatTensor(protein_flatten).to(self.device)

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)

        """Protein vector with DNN."""
        protein_flatten = self.dnn(protein_flatten) 
        protein_flatten = torch.unsqueeze(protein_flatten, 0)

        """Concatenate the two vector and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_flatten), 1)
        for j in range(self.layer_output):
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

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=model.lr, weight_decay=model.weight_decay)

    def train(self, dataset):
        loss_total, trainCorrect, trainPredict = 0, [], [] 
        for data in tqdm.tqdm(dataset):
            self.optimizer.zero_grad()
            predicted = self.model(data[:3])
            loss = F.mse_loss(predicted[0][0].to(torch.float32), data[3].to(torch.float32).to(self.model.device))
            loss_total += loss
            loss.backward()
            self.optimizer.step()
            trainCorrect.append(data[3].to(torch.float32))
            trainPredict.append(predicted[0][0].to(torch.float32))

        trainCorrect = torch.stack(trainCorrect).detach().cpu().numpy()
        trainPredict = torch.stack(trainPredict).detach().cpu().numpy()
        rmse_train = np.sqrt(mean_squared_error(trainCorrect, trainPredict))
        r2_train = r2_score(trainCorrect, trainPredict)
        print('Train RMSE: %.4f , R2: %.4f' %(rmse_train, r2_train))
        torch.cuda.empty_cache()
        return loss_total, rmse_train, r2_train

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        loss_total, testCorrect, testPredict = 0, [], [] 
        for data in tqdm.tqdm(dataset):
            predicted = self.model(data[:3])
            loss = F.mse_loss(predicted[0][0].to(torch.float32), data[3].to(torch.float32).to(self.model.device))
            loss_total += loss
            testCorrect.append(data[3].to(torch.float32))
            testPredict.append(predicted[0][0].to(torch.float32))

        testCorrect = torch.stack(testCorrect).detach().cpu().numpy()
        testPredict = torch.stack(testPredict).detach().cpu().numpy()
        rmse_test = np.sqrt(mean_squared_error(testCorrect, testPredict))
        r2_test = r2_score(testCorrect, testPredict)
        print('Test RMSE: %.4f , R2: %.4f' %(rmse_test, r2_test))
        torch.cuda.empty_cache()
        return loss_total, rmse_test, r2_test


        


