import copy 
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pytorch_lightning as pl
from torch import nn
from torch.nn import Linear, LogSoftmax
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional.classification import binary_auroc, binary_average_precision
from torchmetrics.classification import Accuracy
from torchmetrics.classification import BinaryAUROC
from torch.optim.lr_scheduler import LinearLR
import torch.optim as optim
from sklearn.model_selection import train_test_split
from coco_grape.graph_vectorizer.neural_network.graph_preprocessing import DecompositionalElementVectorizer


class SupervisedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx = self.data[idx]
        node_structure_mtx = torch.from_numpy(node_structure_mtx)
        node_attribute_mtx = torch.from_numpy(node_attribute_mtx)
        edge_structure_mtx = torch.from_numpy(edge_structure_mtx)
        edge_attribute_mtx = torch.from_numpy(edge_attribute_mtx)
        target = torch.tensor(self.targets[idx])
        sample = (node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx, target)
        return sample

class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx = self.data[idx]
        node_structure_mtx = torch.from_numpy(node_structure_mtx)
        node_attribute_mtx = torch.from_numpy(node_attribute_mtx)
        edge_structure_mtx = torch.from_numpy(edge_structure_mtx)
        edge_attribute_mtx = torch.from_numpy(edge_attribute_mtx)
        sample = (node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx)
        return sample



class TransformerEncoder_dim_x_model_dim(pl.LightningModule):
    def __init__(
        self,
        node_structure_input_dim,
        node_attribute_input_dim,
        edge_structure_input_dim,
        edge_attribute_input_dim,
        model_dim=100,
        num_layers=1,
        num_heads=1,
        dim_feedforward=2048,
        output_dim=2,
        dropout=0.1,
        lr=1e-3,
        lr_total_iters=4000,
        lr_start_factor=1e-1,
        problem_type='classification',
        predict_embedding=False):
        super().__init__()
        self.save_hyperparameters()
        
        node_input_dim = self.hparams.node_attribute_input_dim + self.hparams.node_structure_input_dim
        self.node_input_net = nn.Linear(node_input_dim, self.hparams.model_dim)
        
        node_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.node_transformer_encoder = nn.TransformerEncoder(
            node_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        
        feature_node_input_dim = self.hparams.model_dim + self.hparams.node_structure_input_dim
        self.feature_node_input_net = nn.Linear(feature_node_input_dim, self.hparams.model_dim)
        
        feature_node_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.feature_node_transformer_encoder = nn.TransformerEncoder(
            feature_node_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        

        edge_input_dim = self.hparams.edge_attribute_input_dim + self.hparams.edge_structure_input_dim
        self.edge_input_net = nn.Linear(edge_input_dim, self.hparams.model_dim)
        
        edge_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.edge_transformer_encoder = nn.TransformerEncoder(
            edge_encoder_layer, 
            num_layers=self.hparams.num_layers)

        feature_edge_input_dim = self.hparams.model_dim + self.hparams.edge_structure_input_dim
        self.feature_edge_input_net = nn.Linear(feature_edge_input_dim, self.hparams.model_dim)
        
        feature_edge_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.feature_edge_transformer_encoder = nn.TransformerEncoder(
            feature_edge_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        
        output_input_dim = 2 * self.hparams.model_dim * self.hparams.model_dim
        self.output_net = nn.Linear(output_input_dim, self.hparams.output_dim)

        self.problem_type = problem_type
        if self.problem_type == 'classification': self.loss = nn.CrossEntropyLoss()
        elif self.problem_type == 'regression': self.loss = nn.L1Loss()
        else: raise Exception('Unknown problem type:%s' % problem_type)

    def node_edge_encoding(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        #Note: dim=0 is the batch, dim=1 is the elements (e.g. nodes), dim=2 is the element attributes
        #note attributes are concatenated to node decomposition features 
        xn = torch.cat([node_attribute_mtx, node_structure_mtx], 2) #[num_nodes x (num_node_attributes + num_features)]
        #node attributes are embedded with linear layer to model_dim
        xn = self.node_input_net(xn) #[num_nodes x model_dim]
        #a transformer embeds nodes in model_dim
        xn = self.node_transformer_encoder(xn) #[num_nodes x model_dim]
        #edge attributes are concatenated to edge decomposition features 
        xe = torch.cat([edge_attribute_mtx, edge_structure_mtx], 2)
        #edge attributes are embedded with linear layer to model_dim
        xe = self.edge_input_net(xe) #[num_edges x model_dim]
        #a transformer embeds edges in model_dim
        xe = self.edge_transformer_encoder(xe) #[num_edges x model_dim]
        return xn,xe
    
    def encoding_model(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        #Note: dim=0 is the batch, dim=1 is the elements (e.g. nodes), dim=2 is the element attributes
        #nodes and edges are embedded using transformers
        xn,xe = self.node_edge_encoding(node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx) #xn is [num_nodes x model_dim], xe is [num_edges x model_dim]
        
        #compute (node) decomposition feature attributes by summing all node attributes of nodes that are in each feature:
        #consider xn = [num_nodes x model_dim] and node_structure_mtx = [num_nodes x num_features]
        #the sum of all node attributes is: node_structure_mtx.T . xn = [num_features x model_dim]
        xfn = torch.matmul(torch.transpose(node_structure_mtx, 1, 2), xn) #[num_features x model_dim]
        #concatenate one hot encoding identifier for each feature
        fn_onehot_encoding = torch.eye(node_structure_mtx.shape[2])
        fn_onehot_encoding = torch.unsqueeze(fn_onehot_encoding, 0) #add batch index
        fn_onehot_encoding = fn_onehot_encoding.repeat(xfn.shape[0], 1, 1) #repeat to match batch size
        xfn = torch.cat([fn_onehot_encoding, xfn],2) #[num_features x (num_features+model_dim)] #concatenate fn_onehot_encoding to attributes
        #node feature attributes are embedded with linear layer to model_dim
        xfn = self.feature_node_input_net(xfn) #[num_features x model_dim]
        #a transformer embeds features in model_dim
        xfn = self.feature_node_transformer_encoder(xfn) #[num_features x model_dim]
        #combine node embeddings and node decomposition feature embeddings in an [model_dim x model_dim] matrix
        xn1 = torch.matmul(torch.transpose(xn, 1, 2), node_structure_mtx) #[model_dim x num_features]
        xn2 = torch.matmul(xn1, xfn) #[model_dim x model_dim]

        #compute (edge) decomposition feature attributes by summing all edge attributes of edges that are in each feature:
        #consider xe = [num_edges x model_dim] and edge_structure_mtx = [num_edges x num_features]
        #the sum of all edge attributes is: edge_structure_mtx.T . xe = [num_features x model_dim]
        xfe = torch.matmul(torch.transpose(edge_structure_mtx, 1, 2), xe) #[num_features x model_dim]
        #concatenate one hot encoding identifier for each feature
        fe_onehot_encoding = torch.eye(edge_structure_mtx.shape[2])
        fe_onehot_encoding = torch.unsqueeze(fe_onehot_encoding, 0) #add batch index
        fe_onehot_encoding = fe_onehot_encoding.repeat(xfe.shape[0], 1, 1) #repeat to match batch size
        xfe = torch.cat([fe_onehot_encoding, xfe],2) #[num_features x (num_features+model_dim)] #concatenate fn_onehot_encoding to attributes
        #edge feature attributes are embedded with linear layer to model_dim
        xfe = self.feature_edge_input_net(xfe) #[num_features x model_dim]
        #a transformer embeds features in model_dim
        xfe = self.feature_edge_transformer_encoder(xfe) #[num_features x model_dim]
        #combine node embeddings and feature embeddings in an [model_dim x model_dim] matrix
        xe1 = torch.matmul(torch.transpose(xe, 1, 2), edge_structure_mtx) #[model_dim x num_features]
        xe2 = torch.matmul(xe1, xfe) #[model_dim x model_dim]
        
        #combine node and edge information by concatenation
        x = torch.cat([xn2, xe2],2) #[model_dim x 2 model_dim]

        #flatten the rank 2 [model_dim x 2 model_dim] matrix into a rank 1 vector of size 2 model_dim^2
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        x = self.encoding(node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx) #[2 model_dim^2]
        x = self.output_net(x) #[output_dim] where output_dim=2 for binary classification
        return x

    def training_step(self, batch, batch_idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx, y = batch
        y_hat = self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.problem_type == 'classification':
            loss, acc, auroc, auprc = self._shared_eval_step(batch, batch_idx)
            metrics = {"val_acc": acc, "val_loss": loss, "val_auroc":auroc, "val_auprc":auprc}
        elif self.problem_type == 'regression':
            loss = self._shared_eval_step(batch, batch_idx)
            metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics
        
    def test_step(self, batch, batch_idx):
        if self.problem_type == 'classification':
            loss, acc, auroc, auprc = self._shared_eval_step(batch, batch_idx)
            metrics = {"test_acc": acc, "test_loss": loss, "test_auroc":auroc, "val_auprc":auprc}
        elif self.problem_type == 'regression':
            loss = self._shared_eval_step(batch, batch_idx)
            metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx, y = batch
        y_hat = self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        loss = self.loss(y_hat, y)
        if self.problem_type == 'classification':
            acc = binary_accuracy(torch.argmax(y_hat, dim=1), y)
            auroc = binary_auroc(y_hat[:,-1], y)
            auprc = binary_average_precision(y_hat[:,-1], y)
            return loss, acc, auroc, auprc
        elif self.problem_type == 'regression':
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx = batch
        if self.hparams.predict_embedding: return self.encoding(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        else: return self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        #optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        self.lr_scheduler  = LinearLR(optimizer, 
                     start_factor = self.hparams.lr_start_factor, # The number we multiply learning rate in the first epoch
                     total_iters = self.hparams.lr_total_iters) # The number of iterations that multiplicative factor reaches to 1
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        node_structure_input_dim,
        node_attribute_input_dim,
        edge_structure_input_dim,
        edge_attribute_input_dim,
        model_dim=100,
        num_layers=1,
        num_heads=1,
        dim_feedforward=2048,
        output_dim=2,
        dropout=0.1,
        lr=1e-3,
        lr_total_iters=4000,
        lr_start_factor=1e-1,
        problem_type='classification',
        predict_embedding=False):
        super().__init__()
        self.save_hyperparameters()
        
        node_input_dim = self.hparams.node_attribute_input_dim + self.hparams.node_structure_input_dim
        self.node_input_net = nn.Linear(node_input_dim, self.hparams.model_dim)
        
        node_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.node_transformer_encoder = nn.TransformerEncoder(
            node_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        
        feature_node_input_dim = self.hparams.model_dim + self.hparams.node_structure_input_dim
        self.feature_node_input_net = nn.Linear(feature_node_input_dim, self.hparams.model_dim)
        
        feature_node_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.feature_node_transformer_encoder = nn.TransformerEncoder(
            feature_node_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        

        edge_input_dim = self.hparams.edge_attribute_input_dim + self.hparams.edge_structure_input_dim
        self.edge_input_net = nn.Linear(edge_input_dim, self.hparams.model_dim)
        
        edge_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.edge_transformer_encoder = nn.TransformerEncoder(
            edge_encoder_layer, 
            num_layers=self.hparams.num_layers)

        feature_edge_input_dim = self.hparams.model_dim + self.hparams.edge_structure_input_dim
        self.feature_edge_input_net = nn.Linear(feature_edge_input_dim, self.hparams.model_dim)
        
        feature_edge_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.dropout)
        
        self.feature_edge_transformer_encoder = nn.TransformerEncoder(
            feature_edge_encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        output_input_dim = 4 * self.hparams.model_dim
        self.output_net = nn.Linear(output_input_dim, self.hparams.output_dim)

        self.problem_type = problem_type
        if self.problem_type == 'classification': self.loss = nn.CrossEntropyLoss()
        elif self.problem_type == 'regression': self.loss = nn.L1Loss()
        else: raise Exception('Unknown problem type:%s' % problem_type)

    def node_edge_encoding(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        #Note: dim=0 is the batch, dim=1 is the elements (e.g. nodes), dim=2 is the element attributes
        #note attributes are concatenated to node decomposition features 
        xn = torch.cat([node_attribute_mtx, node_structure_mtx], 2) #[num_nodes x (num_node_attributes + num_features)]
        #node attributes are embedded with linear layer to model_dim
        xn = self.node_input_net(xn) #[num_nodes x model_dim]
        #a transformer embeds nodes in model_dim
        xn = self.node_transformer_encoder(xn) #[num_nodes x model_dim]
        #edge attributes are concatenated to edge decomposition features 
        xe = torch.cat([edge_attribute_mtx, edge_structure_mtx], 2)
        #edge attributes are embedded with linear layer to model_dim
        xe = self.edge_input_net(xe) #[num_edges x model_dim]
        #a transformer embeds edges in model_dim
        xe = self.edge_transformer_encoder(xe) #[num_edges x model_dim]
        return xn,xe
    
    def elements_encoding(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        #Note: dim=0 is the batch, dim=1 is the elements (e.g. nodes), dim=2 is the element attributes
        #nodes and edges are embedded using transformers
        xn,xe = self.node_edge_encoding(node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx) #xn is [num_nodes x model_dim], xe is [num_edges x model_dim]
        
        #compute (node) decomposition feature attributes by summing all node attributes of nodes that are in each feature:
        #consider xn = [num_nodes x model_dim] and node_structure_mtx = [num_nodes x num_features]
        #the sum of all node attributes is: node_structure_mtx.T . xn = [num_features x model_dim]
        xfn = torch.matmul(torch.transpose(node_structure_mtx, 1, 2), xn) #[num_features x model_dim]
        #concatenate one hot encoding identifier for each feature
        fn_onehot_encoding = torch.eye(node_structure_mtx.shape[2])
        fn_onehot_encoding = torch.unsqueeze(fn_onehot_encoding, 0) #add batch index
        fn_onehot_encoding = fn_onehot_encoding.repeat(xfn.shape[0], 1, 1) #repeat to match batch size
        xfn = torch.cat([fn_onehot_encoding, xfn],2) #[num_features x (num_features+model_dim)] #concatenate fn_onehot_encoding to attributes
        #node feature attributes are embedded with linear layer to model_dim
        xfn = self.feature_node_input_net(xfn) #[num_features x model_dim]
        #a transformer embeds features in model_dim
        xfn = self.feature_node_transformer_encoder(xfn) #[num_features x model_dim]
        
        #compute (edge) decomposition feature attributes by summing all edge attributes of edges that are in each feature:
        #consider xe = [num_edges x model_dim] and edge_structure_mtx = [num_edges x num_features]
        #the sum of all edge attributes is: edge_structure_mtx.T . xe = [num_features x model_dim]
        xfe = torch.matmul(torch.transpose(edge_structure_mtx, 1, 2), xe) #[num_features x model_dim]
        #concatenate one hot encoding identifier for each feature
        fe_onehot_encoding = torch.eye(edge_structure_mtx.shape[2])
        fe_onehot_encoding = torch.unsqueeze(fe_onehot_encoding, 0) #add batch index
        fe_onehot_encoding = fe_onehot_encoding.repeat(xfe.shape[0], 1, 1) #repeat to match batch size
        xfe = torch.cat([fe_onehot_encoding, xfe],2) #[num_features x (num_features+model_dim)] #concatenate fn_onehot_encoding to attributes
        #edge feature attributes are embedded with linear layer to model_dim
        xfe = self.feature_edge_input_net(xfe) #[num_features x model_dim]
        #a transformer embeds features in model_dim
        xfe = self.feature_edge_transformer_encoder(xfe) #[num_features x model_dim]
         
        out = [xn,xe, xfn, xfe]
        return out

    def encoding(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        xn, xe, xfn, xfe = self.elements_encoding(node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx)
        #xn [num_nodes x model_dim]
        #xe [num_edges x model_dim]
        #xfn [num_features x model_dim]
        #xfe [num_features x model_dim]
        xng = torch.sum(xn, 1) #[1 x model_dim]
        xeg = torch.sum(xe, 1) #[1 x model_dim]
        xfng = torch.sum(xfn, 1) #[1 x model_dim]
        xfeg = torch.sum(xfe, 1) #[1 x model_dim]
        #combine all information by concatenation
        x = torch.cat([xng, xeg, xfng, xfeg],1) #[1 x 4 model_dim]
        return x
        
    def forward(self, node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx):
        x = self.encoding(node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx) #[1 x 4 model_dim]
        x = self.output_net(x) #[output_dim] where output_dim=2 for binary classification
        return x

    def training_step(self, batch, batch_idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx, y = batch
        y_hat = self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.problem_type == 'classification':
            loss, acc, auroc, auprc = self._shared_eval_step(batch, batch_idx)
            metrics = {"val_acc": acc, "val_loss": loss, "val_auroc":auroc, "val_auprc":auprc}
        elif self.problem_type == 'regression':
            loss = self._shared_eval_step(batch, batch_idx)
            metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics
        
    def test_step(self, batch, batch_idx):
        if self.problem_type == 'classification':
            loss, acc, auroc, auprc = self._shared_eval_step(batch, batch_idx)
            metrics = {"test_acc": acc, "test_loss": loss, "test_auroc":auroc, "val_auprc":auprc}
        elif self.problem_type == 'regression':
            loss = self._shared_eval_step(batch, batch_idx)
            metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx, y = batch
        y_hat = self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        loss = self.loss(y_hat, y)
        if self.problem_type == 'classification':
            acc = binary_accuracy(torch.argmax(y_hat, dim=1), y)
            auroc = binary_auroc(y_hat[:,-1], y)
            auprc = binary_average_precision(y_hat[:,-1], y)
            return loss, acc, auroc, auprc
        elif self.problem_type == 'regression':
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx = batch
        if self.hparams.predict_embedding: return self.encoding(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
        else: return self(node_structure_mtx.float(), node_attribute_mtx.float(), edge_structure_mtx.float(), edge_attribute_mtx.float())
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        #optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        self.lr_scheduler  = LinearLR(optimizer, 
                     start_factor = self.hparams.lr_start_factor, # The number we multiply learning rate in the first epoch
                     total_iters = self.hparams.lr_total_iters) # The number of iterations that multiplicative factor reaches to 1
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


class DualElementMatrixTransformerEncoder(object):
    def __init__(self, config,
                 model_name="my_model", 
                 use_tensorboard=True,
                 verbose=True):
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard
        self.model_name = model_name
        self.model = TransformerEncoder(
            node_structure_input_dim=config['node_structure_input_dim'],
            node_attribute_input_dim=config['node_attribute_input_dim'],
            edge_structure_input_dim=config['edge_structure_input_dim'],
            edge_attribute_input_dim=config['edge_attribute_input_dim'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
            lr=config['lr'],
            lr_total_iters=config['lr_total_iters'],
            lr_start_factor=config['lr_start_factor'],
            problem_type=config['problem_type'],
            predict_embedding=True)

    def fit(self, train_loader, val_loader, max_time=None, max_epochs=None):
        if self.use_tensorboard: logger = pl.loggers.TensorBoardLogger("logs", name=self.model_name, default_hp_metric=False)
        else: logger = None
        self.trainer = pl.Trainer(logger=logger, max_time=max_time, max_epochs=max_epochs, log_every_n_steps=1, check_val_every_n_epoch=1, enable_progress_bar=self.verbose, enable_model_summary=self.verbose)
        self.trainer.fit(self.model, train_loader, val_loader)
        return self

    def transform(self, embeddings_dataloader):
        encodings = self.trainer.predict(self.model, embeddings_dataloader)
        embeddings = np.array([encoding.numpy()[0] for encoding in encodings]) #Note: we assume batch_size=1
        return embeddings
    
    #TODO: use elements_encoding to return the encodings of nodes, edges, node_features, edge_features
    def elementwise_transform(self, embeddings_dataloader):
        encodings = self.trainer.predict(self.model, embeddings_dataloader)
        embeddings = np.array([torch.squeeze(encoding,0).numpy() for encoding in encodings]) #dim=0 graphs, dim=1 elements, dim=2 attributes = model_dim
        return embeddings


class DualElementMatrixTransformerEncoderVectorizer(object):
    def __init__(self, 
        config, 
        decomposition_function, 
        nbits=7, 
        attribute_key='vec', 
        parallel=True,
        verbose=True):
        self.verbose = verbose
        self.config = config
        self.decompositional_vectorizer = DecompositionalElementVectorizer(
            decomposition_function=decomposition_function,
            nbits=nbits,  
            attribute_key=attribute_key, 
            parallel=parallel)
        self.dual_element_matrix_transformer_encoder = None

    def fit(self, graphs, targets):
        self.decompositional_vectorizer.fit(graphs)
        tr_train_graphs, tr_val_graphs, tr_train_targets, tr_val_targets = train_test_split(graphs, targets, test_size=.2)
        tr_train_data = self.decompositional_vectorizer.transform(tr_train_graphs)
        tr_val_data = self.decompositional_vectorizer.transform(tr_val_graphs)
        
        train_dataset = SupervisedDataset(tr_train_data, tr_train_targets)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        
        val_dataset = SupervisedDataset(tr_val_data, tr_val_targets)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        
        node_structure_mtx_shape, node_attribute_mtx_shape, edge_structure_mtx_shape, edge_attribute_mtx_shape = self.decompositional_vectorizer.get_data_shape()
        self.config['node_structure_input_dim'] = node_structure_mtx_shape[1]
        self.config['node_attribute_input_dim'] = node_attribute_mtx_shape[1]
        self.config['edge_structure_input_dim'] = edge_structure_mtx_shape[1]
        self.config['edge_attribute_input_dim'] = edge_attribute_mtx_shape[1]
        self.dual_element_matrix_transformer_encoder = DualElementMatrixTransformerEncoder(self.config, model_name=self.config['model_name'], use_tensorboard=self.config['use_tensorboard'], verbose=self.verbose)
        self.dual_element_matrix_transformer_encoder.fit(train_loader, val_loader, max_time=self.config['max_time'], max_epochs=self.config['max_epochs'])
        return self

    def elementwise_transform(self, graphs):
        data = self.decompositional_vectorizer.transform(graphs)
        dataset = UnsupervisedDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])
        embeddings = self.dual_element_matrix_transformer_encoder.elementwise_transform(loader)
        return embeddings
    
    def transform(self, graphs):
        data = self.decompositional_vectorizer.transform(graphs)
        dataset = UnsupervisedDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])
        embeddings = self.dual_element_matrix_transformer_encoder.transform(loader)
        return embeddings

    def fit_transform(self, graphs, targets):
        return self.fit(graphs, targets).transform(graphs)
