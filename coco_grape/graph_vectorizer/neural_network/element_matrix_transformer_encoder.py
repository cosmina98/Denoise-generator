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

from coco_grape.graph_vectorizer.neural_network.graph_preprocessing import DecompositionalNodeVectorizer


class SupervisedDataset(Dataset):
    def __init__(self, data, targets, embeddings=None):
        self.data = data
        self.targets = targets
        self.embeddings = embeddings

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.embeddings is not None: 
            x = torch.from_numpy(self.data[idx])
            x_p = torch.from_numpy(self.embeddings[idx])
            x = torch.cat([x, x_p], 1)
        else: x = torch.from_numpy(self.data[idx])
        sample = (x, torch.tensor(self.targets[idx]))
        return sample

class UnsupervisedDataset(Dataset):
    def __init__(self, data, embeddings=None):
        self.data = data
        self.embeddings = embeddings

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.embeddings is not None: 
            x = torch.from_numpy(self.data[idx])
            x_p = torch.from_numpy(self.embeddings[idx])
            x = torch.cat([x, x_p], 1)
        else: x = torch.from_numpy(self.data[idx])
        return x


class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim=100,
        num_layers=1,
        num_heads=1,
        dim_feedforward=2048,
        output_dim=2,
        transformer_dropout=0.1,
        dropout=0.1,
        lr=1e-3,
        lr_total_iters=4000,
        lr_start_factor=1e-1,
        problem_type='classification',
        predict_embedding=False):
        super().__init__()
        self.save_hyperparameters()

        # self.input_net = nn.Sequential(
        #     nn.Dropout(self.hparams.dropout), 
        #     nn.Linear(self.hparams.input_dim, self.hparams.model_dim), 
        #     nn.LayerNorm(self.hparams.model_dim))
        self.input_net = nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.model_dim, 
            nhead=self.hparams.num_heads, 
            dim_feedforward=self.hparams.dim_feedforward, 
            dropout=self.hparams.transformer_dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.hparams.num_layers)
        
        # self.output_net = nn.Sequential(
        #     nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
        #     nn.LayerNorm(self.hparams.model_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.hparams.dropout),
        #     nn.Linear(self.hparams.model_dim, self.hparams.output_dim),
        # )
        self.output_net = nn.Linear(self.hparams.model_dim, self.hparams.output_dim)

        self.problem_type = problem_type
        if self.problem_type == 'classification': self.loss = nn.CrossEntropyLoss()
        elif self.problem_type == 'regression': self.loss = nn.L1Loss()
        else: raise Exception('Unknown problem type:%s' % problem_type)

    def encoding(self, x):
        x = self.input_net(x)
        x = self.transformer_encoder(x)
        return x
    
    def forward(self, x):
        x = self.encoding(x)
        x = x.sum(1) #dim=0 is the batch, dim=1 is the elements (e.g. nodes), dim=2 is the element attributes
        x = self.output_net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.float())
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
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss(y_hat, y)
        if self.problem_type == 'classification':
            acc = binary_accuracy(torch.argmax(y_hat, dim=1), y)
            auroc = binary_auroc(y_hat[:,-1], y)
            auprc = binary_average_precision(y_hat[:,-1], y)
            return loss, acc, auroc, auprc
        elif self.problem_type == 'regression':
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        if self.hparams.predict_embedding: return self.encoding(x.float())
        else: return self(x.float())
    
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


    def compute_selfattention(self, x, i_layer,d_model,num_heads):
        h = F.linear(x, self.transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=self.transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
        qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
        d_k = q.size()[-1]
        attn_probs = attn_logits / math.sqrt(d_k)
        attn_probs = F.softmax(attn_probs, dim=-1)
        return attn_logits, attn_probs

    def attention(self, x):
        x = self.input_net(x.float())
        attn_logits_maps = []
        attn_probs_maps = []
        num_layers = self.transformer_encoder.num_layers
        d_model = self.transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        norm_first = self.transformer_encoder.layers[0].norm_first
        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = x.clone()
                if norm_first:
                    h = self.transformer_encoder.layers[i].norm1(h)
                attn_logits,attn_probs = self.compute_selfattention(h,i,d_model,num_heads)
                attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
                attn_probs_maps.append(attn_probs)
                # forward of layer i
                x = self.transformer_encoder.layers[i](x)
        return attn_logits_maps,attn_probs_maps


class ElementMatrixTransformerEncoder(object):
    def __init__(self, config,
                 model_name="my_model", 
                 use_tensorboard=True):
        self.use_tensorboard = use_tensorboard
        self.model_name = model_name
        self.model = TransformerEncoder(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            output_dim=config['output_dim'],
            transformer_dropout=config['transformer_dropout'],
            dropout=config['dropout'],
            lr=config['lr'],
            lr_total_iters=config['lr_total_iters'],
            lr_start_factor=config['lr_start_factor'],
            problem_type=config['problem_type'],
            predict_embedding=True)

    def fit(self, train_loader, val_loader, max_time=None, max_epochs=None):
        if self.use_tensorboard: logger = pl.loggers.TensorBoardLogger("logs", name=self.model_name, default_hp_metric=False)
        else: logger = None
        self.trainer = pl.Trainer(logger=logger, max_time=max_time, max_epochs=max_epochs, log_every_n_steps=1, check_val_every_n_epoch=1)
        self.trainer.fit(self.model, train_loader, val_loader)
        return self

    def transform(self, embeddings_dataloader):
        encodings = self.trainer.predict(self.model, embeddings_dataloader)
        embeddings = np.array([encoding.sum(1).numpy()[0] for encoding in encodings]) #Note: we assume batch_size=1
        return embeddings

    def elementwise_transform(self, embeddings_dataloader):
        encodings = self.trainer.predict(self.model, embeddings_dataloader)
        embeddings = np.array([torch.squeeze(encoding,0).numpy() for encoding in encodings]) #dim=0 graphs, dim=1 elements, dim=2 attributes = model_dim
        return embeddings
    
    def tune(self):
        return self

    def attention(self, embeddings_dataloader):
        attn_logits_maps_list = []
        attn_probs_maps_list = []
        for batch in embeddings_dataloader:
            attn_logits_maps, attn_probs_maps = self.model.attention(batch)
            attn_logits_maps_list.append(attn_logits_maps)
            attn_probs_maps_list.append(attn_probs_maps)
        return attn_logits_maps_list, attn_probs_maps_list



class ElementMatrixTransformerEncoderVectorizer(object):
    def __init__(self, 
        config, 
        decomposition_function, 
        nbits=7, 
        node_attribute_key='vec', 
        parallel=True):
        self.config = config
        self.decompositional_vectorizer = DecompositionalNodeVectorizer(
            decomposition_function=decomposition_function,
            nbits=nbits,  
            node_attribute_key=node_attribute_key, 
            parallel=parallel)
        self.element_matrix_transformer_encoder = None

    def fit(self, graphs, targets):
        self.decompositional_vectorizer.fit(graphs)
        tr_train_graphs, tr_val_graphs, tr_train_targets, tr_val_targets = train_test_split(graphs, targets, test_size=.2)
        tr_train_data = self.decompositional_vectorizer.transform(tr_train_graphs)
        tr_val_data = self.decompositional_vectorizer.transform(tr_val_graphs)
        
        train_dataset = SupervisedDataset(tr_train_data, tr_train_targets)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        
        val_dataset = SupervisedDataset(tr_val_data, tr_val_targets)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        
        self.config['input_dim'] = self.decompositional_vectorizer.get_data_shape()[1]
        self.element_matrix_transformer_encoder = ElementMatrixTransformerEncoder(self.config, model_name=self.config['model_name'], use_tensorboard=self.config['use_tensorboard'])
        self.element_matrix_transformer_encoder.fit(train_loader, val_loader, max_time=self.config['max_time'], max_epochs=self.config['max_epochs'])
        return self

    def elementwise_transform(self, graphs):
        data = self.decompositional_vectorizer.transform(graphs)
        dataset = UnsupervisedDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])
        embeddings = self.element_matrix_transformer_encoder.elementwise_transform(loader)
        return embeddings
    
    def transform(self, graphs):
        data = self.decompositional_vectorizer.transform(graphs)
        dataset = UnsupervisedDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])
        embeddings = self.element_matrix_transformer_encoder.transform(loader)
        return embeddings
    
    def annotate(self, graphs, append=True, attribute='vec'):
        embeddings = self.elementwise_transform(graphs)
        out_graphs = []
        for graph, embedding in zip(graphs, embeddings):
            out_graph = graph.copy()
            for node_idx, attribute_vec in zip(graph.nodes(), embedding):
                if append and attribute in out_graph.nodes[node_idx]: out_graph.nodes[node_idx][attribute] = np.hstack([out_graph.nodes[node_idx][attribute].flatten(),attribute_vec.flatten()])
                else: out_graph.nodes[node_idx][attribute] = attribute_vec
            out_graphs.append(out_graph)
        return out_graphs

    def fit_transform(self, graphs, targets):
        return self.fit(graphs, targets).transform(graphs)

    def attention(self, graphs):
        data = self.decompositional_vectorizer.transform(graphs)
        dataset = UnsupervisedDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config['num_workers'])
        attn_logits_maps_list, attn_probs_maps_list = self.element_matrix_transformer_encoder.attention(loader)
        return attn_logits_maps_list, attn_probs_maps_list

    def feature_encoding(self):
        #returns the matrix n_features x model_dim: i.e. each row is a model_dim vector representing the row id feature
        return self.element_matrix_transformer_encoder.model.input_net[1].weight.detach().numpy().T

