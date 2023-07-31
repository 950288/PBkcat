import torch
import torch.nn as nn
import pickle
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error,r2_score
import tqdm

num_filters = 32
kernel_size = 3

class KcatPrediction(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.dim: int = args['dim']
        self.layer_output: int = args['layer_output']
        self.layer_gnn: int = args['layer_gnn']
        self.layer_dnn: int = args['layer_dnn']
        self.lr = args['lr']
        self.len_fingerprint: int = args['len_fingerprint']
        self.weight_decay = args['weight_decay']
        self.embed_fingerprint = nn.Embedding(self.len_fingerprint, self.dim)
        self.W_gnn = nn.ModuleList([
            nn.Linear(self.dim, self.dim)
            for _ in range(self.layer_gnn)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(kernel_size, 26)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=(kernel_size, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        ])
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(59328, 512), # 64*394=25216
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim)
        ])
        self.dnns = nn.ModuleList([  
            nn.Linear(96564, 512),
            nn.Linear(512, 256),
            # for _ in range(self.layer_dnn - 1)
        ]).append(nn.Linear(256, self.dim))
        
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

    def _get_conv_output_size(self, input_size, num_filters, kernel_size):
        # Calculate the output size after passing through convolutional layers
        # Assuming padding is 0 and stride is 1
        conv_output_size = input_size[1] - kernel_size + 1
        conv_output_size = (conv_output_size - kernel_size + 1) // 2
        conv_output_size *= num_filters * 2
        return conv_output_size
    
    def forward(self, inputs):

        fingerprints, adjacency, protein = inputs
        fingerprints = torch.LongTensor(fingerprints).to(self.device)
        adjacency = torch.FloatTensor(adjacency).to(self.device)
        protein = torch.FloatTensor(protein).to(self.device)

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)

        """Protein vector with CNN."""
        # print(protein.shape)
        protein = protein.unsqueeze(0)
        # print(protein.shape)
        for layer in self.conv_layers:
            protein = layer(protein)
            # print(protein.shape)
        protein = protein.view(protein.size(0), -1)
        # print(protein.shape)

        protein_flatten = torch.flatten(protein)
        # print(protein_flatten.shape)

        for layer in self.fc_layers:
            protein_flatten = layer(protein_flatten)
            # print(protein_flatten.shape)
        protein_flatten = protein_flatten.unsqueeze(0)

        """Concatenate the two vector and output the interaction."""
        # print(compound_vector.shape, protein_flatten.shape)
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
        random.shuffle(dataset)        
        self.model.train()
        for data in tqdm.tqdm(dataset):
            self.optimizer.zero_grad()
            predicted = self.model(data[:3])
            loss = F.mse_loss(predicted[0][0].to(torch.float32), data[3].to(torch.float32).to(self.model.device))
            loss_total += loss.item()
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
        self.model.eval()
        for data in tqdm.tqdm(dataset):
            predicted = self.model(data[:3])
            loss = F.mse_loss(predicted[0][0].to(torch.float32), data[3].to(torch.float32).to(self.model.device))
            loss_total += loss.item()
            testCorrect.append(data[3].to(torch.float32))
            testPredict.append(predicted[0][0].to(torch.float32))

        testCorrect = torch.stack(testCorrect).detach().cpu().numpy()
        testPredict = torch.stack(testPredict).detach().cpu().numpy()
        rmse_test = np.sqrt(mean_squared_error(testCorrect, testPredict))
        r2_test = r2_score(testCorrect, testPredict)
        print('Test RMSE: %.4f , R2: %.4f' %(rmse_test, r2_test))
        torch.cuda.empty_cache()
        return loss_total, rmse_test, r2_test


        


