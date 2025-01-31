import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import scipy
from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder
from utils import sparse_mx_to_torch_sparse_tensor, normalize, symmetrize, remove_self_loop, sparse_tensor_add_self_loop, adj_values_one

warnings.simplefilter("ignore")
EOS = 1e-10


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    rowsum = features.sum(dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1,1)
    features = features * r_inv
    return features


def load_mag():

    path = "./data/mag-4/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)

    feat = torch.load(path+'feat.pt').float()
    nnodes = len(feat)
    adj_1 = torch.load(path+'pap.pt').coalesce()
    adj_2 = torch.load(path+'pp.pt').coalesce()
    adjs = [adj_1, adj_2]
    adjs = [sparse_tensor_add_self_loop(adj).coalesce() for adj in adjs]
    adjs = [adj_values_one(adj) for adj in adjs]
    adjs = [normalize(adj, mode='sym', sparse=True) for adj in adjs]

    train_index = torch.load(path+'train_index.pt')
    val_index = torch.load(path+'val_index.pt')
    test_index = torch.load(path+'test_index.pt')

    train_mask = torch.zeros(nnodes, dtype=torch.int64)
    train_mask[train_index] = 1
    val_mask = torch.zeros(nnodes, dtype=torch.int64)
    val_mask[val_index] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.int64)
    test_mask[test_index] = 1

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_dblp():
    path = "data/dblp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')


    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_acm():
    path = "./data/acm/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_yelp():
    path = "./data/yelp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = F.normalize(feat, p=2, dim=1)
    adjs = torch.load(path+'adj.pt')

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_amazon():
    path = "./data/amazon/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = feat.type(torch.float32)
    feat = F.normalize(feat, p=2, dim=1)
    adj1 = torch.load(path+'upu.pt')
    adj2 = torch.load(path+'usu.pt')
    adj3 = torch.load(path+'uvu.pt')
    adjs = [adj1,adj2,adj3]
    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')
    adjs = [normalize(adj, mode='sym',sparse=True) for adj in adjs]
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def dense_to_sparse_pytorch(adjacency_matrix):
    indices = adjacency_matrix.nonzero()
    values = adjacency_matrix[indices[:, 0], indices[:, 1]]
    
    sparse_matrix = torch.sparse_coo_tensor(indices.t(), values, adjacency_matrix.size())
    return sparse_matrix

def normalize_adj_row(adj):
    print(1)
    row_sum = adj.sum(1)  
    r_inv = torch.pow(row_sum, -1)  
    r_inv[torch.isinf(r_inv)] = 0.  
    r_mat_inv = torch.diag(r_inv)  
    return r_mat_inv @ adj  
def normalize_adj_row_sparse(adj):
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()  
    r_inv = row_sum.pow(-1)  
    r_inv[torch.isinf(r_inv)] = 0.  
    
    r_mat_inv = torch.diag(r_inv)
    
    normalized_adj = r_mat_inv @ adj.to_dense()
    
    return normalized_adj.to_sparse()
import scipy.io
def load_esp():
    folder_path = "./data/esp-game/"
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    # print("L:",L.keys())
    # print("txt:",txt.keys())
    # print("img",img.keys())
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data= torch.tensor(txt_data, dtype=torch.float32)  
    img_data = torch.tensor(img_data, dtype=torch.float32)  
    txt_data = F.normalize(txt_data, p=2, dim=1)
    img_data = F.normalize(img_data, p=2, dim=1)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)  
    # print("Label shape:", label.shape)  # 
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)  
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    # print("View feature",view_feature)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_flick():
    folder_path = "./data/flickr30k/"
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data= torch.tensor(txt_data, dtype=torch.float32)  
    img_data = torch.tensor(img_data, dtype=torch.float32) 
    txt_data = F.normalize(txt_data, p=2, dim=1)
    img_data = F.normalize(img_data, p=2, dim=1)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)  
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)  
    # print("Label shape:", label.shape)  
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)  
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    # print("View feature",view_feature)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_iap():
    folder_path = "./data/iaprtc12/"
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data= torch.tensor(txt_data, dtype=torch.float32)  
    img_data = torch.tensor(img_data, dtype=torch.float32)  
    txt_data = F.normalize(txt_data, p=2, dim=1)
    img_data = F.normalize(img_data, p=2, dim=1)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)  
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)  
    # print("Label shape:", label.shape)
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)  
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_nus():
    folder_path = "./data/nus-wide/"
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data= torch.tensor(txt_data, dtype=torch.float32)  
    img_data = torch.tensor(img_data, dtype=torch.float32)  
    txt_data = F.normalize(txt_data, p=2, dim=1)
    img_data = F.normalize(img_data, p=2, dim=1)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)  
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)  
    # print("Label shape:", label.shape)  
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)  
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs
import numpy as np
def create_masks(label_array, train_ratio=0.5, val_ratio=0.25, random_seed=42):
    np.random.seed(random_seed) 
    data_length = len(label_array)
    unique_labels = np.unique(label_array)
    train_mask = np.zeros(data_length, dtype=bool)
    val_mask = np.zeros(data_length, dtype=bool)
    test_mask = np.zeros(data_length, dtype=bool)

    for label in unique_labels:
        label_indices = np.where(label_array == label)[0]
        n_label_samples = len(label_indices)

        train_size = int(train_ratio * n_label_samples)
        val_size = int(val_ratio * n_label_samples)
        test_size = n_label_samples - train_size - val_size  
        indices = np.random.permutation(label_indices)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask
def create_zero_adjacency_matrix(feature_shape):
    num_samples = feature_shape[0]
    indices = torch.empty((2, 0), dtype=torch.int64) 
    values = torch.empty(0, dtype=torch.float32)  
    size = (num_samples, num_samples)  
    adjacency_matrix = torch.sparse.FloatTensor(indices, values, size)
    return adjacency_matrix
def load_data(args):
    if args.dataset == 'dblp':
        return load_dblp()
    elif args.dataset == 'mag':
        return load_mag()
    elif args.dataset == 'acm':
        return load_acm()
    elif args.dataset == 'yelp':
        return load_yelp()
    elif args.dataset == 'amazon':
        return load_amazon()
    elif args.dataset == 'esp':
        return load_esp()
    elif args.dataset == 'flickr':
        return load_flick()
    elif args.dataset == 'iapr':
        return load_iap()
    elif args.dataset == 'nus':
        return load_nus()
    
if __name__ == '__main__':
    load_esp()
