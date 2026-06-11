import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5

def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        h_e = self.encoder(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-var can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-var and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)

class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()

        self.decoder = decoder_net
        self.distribution = distribution
        self.num_vals=num_vals

    def decode(self, z):
        h_d = self.decoder(z)

        if self.distribution == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]

        elif self.distribution == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            return [mu_d]
        
        else:
            raise ValueError('Either `categorical` or `bernoulli`')

    def sample(self, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            mu_d = outs[0]
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            x_new = torch.bernoulli(mu_d)
        
        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new

    def log_prob(self, x, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)
            
        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
            
        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)

class Prior(nn.Module):
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)

class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_vals=256, L=16, likelihood_type='categorical', beta=4):
        super(VAE, self).__init__()
        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(distribution=likelihood_type, decoder_net=decoder_net, num_vals=num_vals)
        self.prior = Prior(L=L)
        self.num_vals = num_vals
        self.likelihood_type = likelihood_type
        self.beta = beta

    def forward(self, x, reduction='avg'):
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        RE = self.decoder.log_prob(x, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)

        if reduction == 'sum':
            return -(RE + self.beta * KL).sum()
        else:
            return -(RE + self.beta * KL).mean()

    def sample_latent(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return z

    def sample(self, batch_size=64):
        z = self.sample_latent(batch_size)
        x = self.decoder.sample(z)
        return x



class VAETransformer(object):
    def __init__(self, dim_input, dim_latent=32, dim_hidden=800, n_layers=1, num_discretization_levels=8, beta=1, num_epochs=1000, lr=1e-3, max_patience=10, verbose=False):
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.n_components = dim_latent
        self.dim_hidden = dim_hidden
        self.num_vals = num_discretization_levels
        self.verbose = verbose
        self.val_size = 0.2
        self.batch_size = 32
        self.n_layers = n_layers
        self.beta = beta
        self.lr = lr # learning rate
        self.num_epochs = num_epochs
        self.max_patience = max_patience # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
        
        D = self.dim_input   # input dimension
        L = self.dim_latent  # latent dimension
        M = self.dim_hidden  # hidden dimension

        layers = [nn.Linear(D, M), nn.LeakyReLU()]
        for i in range(self.n_layers):
            layers.extend([nn.Linear(M, M), nn.LeakyReLU()])
        layers.append(nn.Linear(M, 2 * L))
        encoder_net = nn.Sequential(*layers)

        layers = [nn.Linear(L, M), nn.LeakyReLU()]
        for i in range(self.n_layers):
            layers.extend([nn.Linear(M, M), nn.LeakyReLU()])
        layers.append(nn.Linear(M, (self.num_vals+1) * D))
        decoder_net = nn.Sequential(*layers)

        prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))
        self.model = VAE(encoder_net=encoder_net, decoder_net=decoder_net, num_vals=self.num_vals+1, L=L, beta=self.beta)
        self.optimizer = torch.optim.Adamax([p for p in self.model.parameters() if p.requires_grad == True], lr=self.lr)

    def evaluation(self, test_loader, epoch=None):
        self.model.eval()
        loss = 0.
        N = 0.
        for indx_batch, test_batch in enumerate(test_loader):
            loss_t = self.model.forward(test_batch, reduction='sum')
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
        loss = loss / N
        if self.verbose:
            print(f'Epoch: {epoch}, val nll={loss}')
        return loss

    def transform_data(self, X):
        X = (X - self.min)/(self.max - self.min)
        X = (X*self.num_vals).astype(int).astype(np.float32)
        X = torch.from_numpy(X)
        return X

    def inverse_transform_data(self, X):
        X = X.detach().numpy()
        X = X / self.num_vals
        X = (self.max - self.min) * X + self.min
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def fit(self, X, y=None):
        self.max = np.max(X)
        self.min = np.min(X)
        X = self.transform_data(X)
        train_data, val_data = train_test_split(X, test_size=self.val_size)
        training_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        nll_val = []
        best_nll = 1000.
        patience = 0

        for e in range(self.num_epochs):
            self.model.train()
            for indx_batch, batch in enumerate(training_loader):
                loss = self.model.forward(batch)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            # Validation
            loss_val = self.evaluation(val_loader, epoch=e)
            if e == 0:
                best_nll = loss_val
            else:
                if loss_val < best_nll:
                    best_nll = loss_val
                    patience = 0
                else:
                    patience = patience + 1
            if patience > self.max_patience:
                break
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        #return prob
        X = self.transform_data(X)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        log_probs = []
        for batch in test_loader:
            mu_e, log_var_e = self.model.encoder.encode(batch)
            z = self.model.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
            log_prob = self.model.decoder.log_prob(batch, z)
            log_probs.append(log_prob.detach().numpy())
        log_probs = np.vstack(log_probs)
        return log_probs
    
    def reconstruct(self, X):
        self.model.eval()
        #return latent
        X = self.transform_data(X)
        mu_e, log_var_e = self.model.encoder.encode(X)
        Z = self.model.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        X_new = self.model.decoder.sample(Z)
        X_new = self.inverse_transform_data(X_new)
        return X_new
    
    def transform(self, X):
        self.model.eval()
        #return latent
        X = self.transform_data(X)
        mu_e, log_var_e = self.model.encoder.encode(X)
        Z = self.model.encoder.sample(mu_e=mu_e, log_var_e=log_var_e).detach().numpy()
        return Z

    def inverse_transform(self, Z):
        self.model.eval()
        Z = torch.from_numpy(Z.astype(np.float32))
        X_new = self.model.decoder.sample(Z)
        X_new = self.inverse_transform_data(X_new)
        return X_new

    def sample_latent(self, n_samples=1):
        self.model.eval()
        Z = self.model.sample_latent(n_samples)
        return Z

    def sample(self, n_samples=1):
        self.model.eval()
        X = self.model.sample(n_samples)
        X = self.inverse_transform_data(X)
        return X
