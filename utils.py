import torch
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AE(nn.Module):
    """Transformer-based autoencoder used for representation learning."""

    def __init__(self, input_dim: int):
        super(AE, self).__init__()

        # Determine model width based on the feature dimension while keeping it GPU friendly.
        nearest_power_of_2 = 2 ** math.ceil(math.log2(max(16, input_dim)))
        self.d_model = min(256, nearest_power_of_2)

        # Ensure the embedding dimension is divisible by the number of heads.
        self.nhead = 8 if self.d_model >= 64 else 4 if self.d_model >= 32 else 2
        self.seq_len = input_dim
        self.latent_dim = max(self.d_model // 2, 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=False,
            activation='gelu',
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=False,
            activation='gelu',
        )
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.positional_encoding = PositionalEncoding(self.d_model)
        self.target_positional_encoding = PositionalEncoding(self.d_model)

        self.input_projection = nn.Linear(1, self.d_model)
        self.latent_projection = nn.Linear(self.d_model, self.latent_dim)
        self.memory_projection = nn.Linear(self.latent_dim, self.d_model)
        self.output_projection = nn.Linear(self.d_model, 1)

        # Learnable queries used by the decoder to reconstruct the original feature sequence.
        self.decoder_queries = nn.Parameter(torch.zeros(self.seq_len, self.d_model))
        nn.init.normal_(self.decoder_queries, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        # Prepare source tokens.
        src = x.unsqueeze(-1)  # (batch, seq_len, 1)
        src = self.input_projection(src)  # (batch, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch, d_model)
        src = self.positional_encoding(src)

        # Encoder
        memory = self.encoder_transformer(src)  # (seq_len, batch, d_model)

        pooled_memory = memory.mean(dim=0)  # (batch, d_model)
        latent = self.latent_projection(pooled_memory)  # (batch, latent_dim)

        # Blend the latent representation back into the encoder memory before decoding.
        decoder_memory = self.memory_projection(latent).unsqueeze(0)
        memory = memory + decoder_memory

        tgt = self.decoder_queries.unsqueeze(1).expand(-1, x.size(0), -1)
        tgt = self.target_positional_encoding(tgt)

        decoded = self.decoder_transformer(tgt, memory)
        decoded = decoded.transpose(0, 1)  # (batch, seq_len, d_model)
        recon = self.output_projection(decoded).squeeze(-1)  # (batch, seq_len)

        return latent, recon

class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]
        
        ## This is the denominator for InfoNCE loss: ONE time of traversal
        # sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        ## This is the denominator for our proposed CRC loss: TWO times of traversal
        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal))
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
def score_detail(y_test,y_test_pred,if_print=False):
    # Confusion matrix
    if if_print == True:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        # Accuracy 
        print('Accuracy ',accuracy_score(y_test, y_test_pred))
        # Precision 
        print('Precision ',precision_score(y_test, y_test_pred))
        # Recall
        print('Recall ',recall_score(y_test, y_test_pred))
        # F1 score
        print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2))

def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model, get_confidence=False, en_or_de=False):
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, indcies = torch.sort(F.cosine_similarity(train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_normal, indcies = torch.sort(F.cosine_similarity(train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_abnormal, indcies = torch.sort(F.cosine_similarity(train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))

    values_features_all = values_features_all.cpu().detach().numpy()

    values_features_test = F.cosine_similarity(test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, indcies = torch.sort(F.cosine_similarity(train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_normal, indcies = torch.sort(F.cosine_similarity(train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_abnormal, indcies = torch.sort(F.cosine_similarity(train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

    values_recon_all = values_recon_all.cpu().detach().numpy()

    values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy())

    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_features_all,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x # Estimated parameter values

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()

    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2-pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy())

    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x # Estimated parameter values

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()

    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4-pdf3)).cpu().detach().numpy().astype("float32")

    if not isinstance(y_test, int):
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)

    y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
    
    if not isinstance(y_test, int):
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
        return result_encoder, result_decoder, result_final
    else:
        return y_test_pred_no_vote
