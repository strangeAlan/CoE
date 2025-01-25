# -*- coding: utf-8 -*-
import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from data_loader import load_data
from model import GCN, GCL, AGG,MODEL,MixGNN
from graph_learners import *
from utils import *
from params import *
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans as KMeans_py
from sklearn.metrics import f1_score
from Cooperation import getCooperationResult
import random

EOS = 1e-10
args = set_params()

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.training = False

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def loss_cls_Main(self, model, mask, view_features, labels,learned_fused_adj,learned_specific_adjs,concat_feature=None):
        # logits = model(features,shared_adjs,specific_adjs)
        logits = model(view_features,learned_fused_adj,learned_specific_adjs,concat_feature)
        loss = 0
        accurs = []
        for i in range (model.num_g + 1):
            # logp = F.log_softmax(logits[i], 1)
            p = F.softmax(logits[i], 1)
            single_loss = model.criterion(p[mask], labels[mask])
            accu = accuracy(p[mask], labels[mask])
            loss += single_loss
            accurs.append(accu)
        loss = loss/(model.num_g +1)
        return loss

    def loss_cls(self, model, mask, features, labels):
        torch.cuda.empty_cache()
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu
    
    def train_one_classfier(self, Adj, concat_feature, nfeats, labels, nclasses, train_mask, val_mask, args):
        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)
        bad_counter = 0
        best_val = 0
        best_model = None
        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            # test_mask = test_mask.cuda()
            # features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, train_accu = self.loss_cls(model, train_mask, concat_feature, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                val_loss, val_accu = self.loss_cls(model, val_mask, concat_feature, labels)
                if val_accu > best_val:
                    bad_counter = 0
                    best_val = val_accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1
                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        logits = best_model(concat_feature)#
        # logp = F.log_softmax(logits, 1)
        p = F.softmax(logits,1)
        p_test,p_train = p[val_mask],p[train_mask]
        return p_test,p_train
    
    def loss_Main(self, model, specific_graph_learner, fused_graph_learner, concat_feature, view_features, adjs,
                 optimizer, mask,labels):
        optimizer.zero_grad()

        learned_specific_adjs = []
    
        for i in range(len(adjs)):
            specific_adjs_embedding = specific_graph_learner[i](view_features[i])
            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)
            
        fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))
        learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)
        z_specific_adjs = [model.embedding(view_features[i], learned_specific_adjs[i]) for i in range(len(adjs))]
        z_fused_adj = model.embedding(concat_feature, learned_fused_adj,concat=True)

        if args.contrast_batch_size:
            node_idxs = list(range(concat_feature.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss1 = 0
            for batch in batches:
                weight = len(batch) / concat_feature.shape[0]
                loss1 += model.cal_loss([z[batch] for z in z_specific_adjs], z_fused_adj[batch]) * weight
        else:
            loss1 = model.cal_loss(z_specific_adjs, z_fused_adj)
        loss2 = self.loss_cls_Main(model, mask, view_features, labels,learned_fused_adj,learned_specific_adjs,concat_feature)
        if args.contrast_batch_size:
            node_idxs = list(range(concat_feature.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss3 = 0
            for batch in batches:
                weight = len(batch) / concat_feature.shape[0]
                loss3 += model.cal_sharedloss([z[batch] for z in z_specific_adjs]) * weight
        else:
            loss3 = model.cal_sharedloss(z_specific_adjs)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        return loss

    def loss_MixGNN(self, model,features, view_features,optimizer, mask,labels):
        optimizer.zero_grad()
        learned_specific_adjs = []
        indexes_to_extract = model.indexes + [-1]  
        selected_features = [view_features[idx] for idx in indexes_to_extract]
        for i in range(model.mixNum):
            specific_adjs_embedding = model.specific_graph_learner[i](selected_features[i])
            learned_specific_adj = model.specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)
        fused_embedding = model.fused_graph_learner(torch.cat(selected_features, dim=1))
        learned_fused_adj = model.fused_graph_learner.graph_process(fused_embedding)
        z_specific_adjs = [model.embedding(features, learned_specific_adjs[i]) for i in range(model.mixNum)]
        z_fused_adj = model.embedding(features, learned_fused_adj)

        # loss1 = model.cal_loss(z_specific_adjs, z_fused_adj)
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss1 = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss1 += model.cal_loss([z[batch] for z in z_specific_adjs], z_fused_adj[batch]) * weight
        else:
            loss1 = model.cal_loss(z_specific_adjs, z_fused_adj)
        loss2 = self.loss_cls_Main(model, mask, features, labels,learned_fused_adj,learned_specific_adjs)
        # loss3 = model.cal_sharedloss(z_specific_adjs)
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss3 = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss3 += model.cal_sharedloss([z[batch] for z in z_specific_adjs]) * weight
        else:
            loss3 = model.cal_sharedloss(z_specific_adjs)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        return loss

    def acc(self,p,labels,mask):
        test_accu = accuracy(p, labels[mask])
        preds = torch.argmax(p, dim=1)
        test_f1_macro = f1_score(labels[mask].cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(labels[mask].cpu(), preds.cpu(), average='micro')
        return test_accu,test_f1_macro,test_f1_micro

    def train(self, args):
        print(args)
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        view_features_orginal, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adjs_original = load_data(args)
        for trial in range(1):
            self.setup_seed(args.seed)
            adjs = copy.deepcopy(adjs_original)
            view_features = view_features_orginal
            concat_feature = torch.cat(view_features, dim=1)
            specific_graph_learner = [ATT_learner(2, nfeats, args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner) for _ in range(len(adjs))]
            fused_graph_learner = ATT_learner(2, nfeats*(len(adjs)), args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner)
            model = MODEL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, sparse=args.sparse, num_g=len(adjs),
                         in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                         dropout_cls=args.dropout_cls, dropout_adj=args.dropedge_cls,generalize=True)
            optimizer = torch.optim.Adam([{'params': specific_graph_learner[i].parameters()} for i in range(len(adjs))] +
                                        [{'params': fused_graph_learner.parameters()}] +
                                        [{'params': model.parameters()}], lr=args.lr, weight_decay=args.w_decay)
            if len(adjs)>2:
                MixGNN_List= []
                MixGNN_optimizers = []  
                for i in range(len(adjs)):
                    for j in range(i+1,len(adjs)):
                        id = [i,j]
                        m = MixGNN(nfeats,2,len(adjs),nclasses,args,id)
                        MixGNN_List.append(m)
                        op = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.w_decay)
                        MixGNN_optimizers.append(op)
            else:
                MixGNN_List= None
                MixGNN_optimizers = None
            if torch.cuda.is_available():
                model = model.to(device)
                specific_graph_learner = [m.to(device) for m in specific_graph_learner]
                fused_graph_learner = fused_graph_learner.to(device)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)
                # features = features.to(device)
                labels = labels.to(device)
                adjs = [adj.to(device) for adj in adjs]
                view_features = [f.to(device) for f in view_features]
                concat_feature = concat_feature.to(device)
                # if len(adjs)>2:
                #     for m in MixGNN_List:
                #         m.to(device)
                #         m.specific_graph_learner = [m2.to(device) for m2 in m.specific_graph_learner]
                #         m.fused_graph_learner = m.fused_graph_learner.to(device)
            best_model = None
            best_specific_graph_learner = None
            best_fused_graph_learner = None
            best_MixGNN_List = None
            best_val = 0
            patience = args.patience
            bad_accounter =0
            for epoch in range(1, args.epochs + 1):
                model.train()
                [learner.train() for learner in specific_graph_learner]
                fused_graph_learner.train()
                self.training = True
                loss = self.loss_Main(model, specific_graph_learner, fused_graph_learner, concat_feature, view_features, adjs,
                                    optimizer, train_mask, labels)
                print("Trail {:05d} | Epoch {:05d} | CL Loss {:.4f}".format(trial+1 , epoch, loss.item()))
                # if len(adjs)>2:
                #     for i, m in enumerate(MixGNN_List):
                #         m.train()
                #         [learner.train() for learner in m.specific_graph_learner]
                #         m.fused_graph_learner.train()

                #         self.loss_MixGNN(m,features, view_features,MixGNN_optimizers[i], train_mask, labels)
                if epoch % args.eval_freq == 0:
                    # start to eval
                    model.eval()
                    [learner.eval() for learner in specific_graph_learner]
                    fused_graph_learner.eval()
                    if len(adjs)>2:
                        for i, m in enumerate(MixGNN_List):
                            m.eval()
                            [learner.eval() for learner in m.specific_graph_learner]
                            m.fused_graph_learner.eval()
                    self.training = False
                    val_accuracy_values,model, specific_graph_learner,fused_graph_learner,MixGNN_List = self.test(args,model,specific_graph_learner,fused_graph_learner,MixGNN_List,adjs,view_features,
                                                    concat_feature,nfeats,labels,nclasses, train_mask, val_mask,device)
                    if val_accuracy_values[-1] > best_val:
                            best_model = model
                            best_specific_graph_learner = specific_graph_learner
                            best_fused_graph_learner = fused_graph_learner
                            best_MixGNN_List = MixGNN_List
                            best_val = val_accuracy_values[-1]
                            bad_accounter = 0
                            
                    else :
                        bad_accounter+=1
                        if(bad_accounter>patience):
                            break

        print("-----------------------------Testing---------------------------------")
        test_accuracy_values,_, _,_,_ = self.test(args,best_model,best_specific_graph_learner,best_fused_graph_learner,best_MixGNN_List,adjs,view_features,
                                                         concat_feature,nfeats,labels,nclasses, train_mask, test_mask,device)
        print("--------------Final Test Result-------------")
        # data = pd.read_csv(f'dataset/{args.dataset}_test_result.csv')
        print("dataset:",args.dataset)
        print("Final acc:",test_accuracy_values[-1])
        print("--------------------------------------------")
        # result = pd.DataFrame({ 'dataset':[args.dataset],
        #                     'Accuracy': [test_accuracy_values[-1]]
        #                     })
        # try:
        #     df = pd.read_csv(f'result/{args.dataset}_test_result.csv')
        #     max_score = df['Accuracy'].iloc[0]
        # except FileNotFoundError:
        #     max_score = 0
        # if test_accuracy_values[-1]>max_score:
        #     print("get better result!max_acc {}---------------> {}".format(max_score,test_accuracy_values[-1]))
        #     print("-----------saving configs---------------------")
        #     result.to_csv(f'result/{args.dataset}_test_result.csv', index=False)
        #     save_args_to_file(args, f'args/{args.dataset}_args.json')
        #     print("done")

    def test(self,args,model,specific_graph_learner,fused_graph_learner,MixGNN_List,adjs,view_features,concat_feature,nfeats,labels,nclasses, train_mask, test_mask,device):
         # start to eval
                    model.eval()
                    [learner.eval() for learner in specific_graph_learner]
                    fused_graph_learner.eval()
                    if len(adjs)>2:
                        for i, m in enumerate(MixGNN_List):
                            m.eval()
                            [learner.eval() for learner in m.specific_graph_learner]
                            m.fused_graph_learner.eval()
                    self.training = False
                    if args.downstream_task == 'classification':
                        matrix = []
                        train_matrix = []
                        test_accurs =[]
                        test_f1_mas =[]
                        test_f1_mis =[]
                        train_accurs = []
                        train_f1_mas =[]
                        train_f1_mis =[]
                    
                        # 
                        fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))#fused
                        learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)
                        if args.sparse:
                            learned_fused_adj.edata['w'] = learned_fused_adj.edata['w'].detach()
                        else:
                            learned_fused_adj = learned_fused_adj.detach()
                        learned_specific_adjs = []
                        for i in range(len(adjs)):
                            specific_adjs_embedding = specific_graph_learner[i](view_features[i])#specific
                            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
                            if args.sparse:
                                learned_specific_adj.edata['w'] = learned_specific_adj.edata['w'].detach()
                            else:
                                learned_specific_adj = learned_specific_adj.detach()
                            learned_specific_adjs.append(learned_specific_adj)  
                        # 
                        for i in range(len(adjs)+1):
                            if args.sparse == False:
                                if i !=len(adjs):
                                    adj = learned_specific_adjs[i]+learned_fused_adj
                                else:
                                    adj = learned_fused_adj
                            else:
                                if i !=len(adjs):
                                    adj = add_graphs(learned_specific_adjs[i], learned_fused_adj)
                                else :
                                    adj = learned_fused_adj
                            adj = adj
                            predict,train_predict = self.train_one_classfier(adj,concat_feature,concat_feature.shape[1],labels,nclasses, train_mask,test_mask, args)
                            predict_transposed = predict.transpose(0, 1)
                            matrix.append(predict_transposed)
                            train_predict_transposed=train_predict.transpose(0, 1)
                            train_matrix.append(train_predict_transposed)
                            # 
                            test_accur,test_f1_ma,test_f1_mi = self.acc(predict,labels,test_mask)
                            test_accurs.append(test_accur)
                            test_f1_mas.append(test_f1_ma)
                            test_f1_mis.append(test_f1_mi)
                            # 
                            train_acc,train_f1_ma,train_f1_mi = self.acc(train_predict,labels,train_mask)
                            train_accurs.append(train_acc)
                            train_f1_mas.append(train_f1_ma)
                            train_f1_mis.append(train_f1_mi)
                          
                        # ...
                        predictions = model(view_features,learned_fused_adj,learned_specific_adjs,concat_feature)
                        predictions = [F.softmax(prediction,1) for prediction in predictions]
                        for i in range(len(predictions)):
                            # 
                            train_predict =predictions[i][train_mask]
                            test_predict = predictions[i][test_mask]
                            predict_transposed=test_predict.transpose(0, 1)
                            matrix.append(predict_transposed)
                            train_predict_transposed=train_predict.transpose(0, 1)
                            train_matrix.append(train_predict_transposed)
                            # 
                            test_accur,test_f1_ma,test_f1_mi = self.acc(test_predict,labels,test_mask)
                            test_accurs.append(test_accur)
                            test_f1_mas.append(test_f1_ma)
                            test_f1_mis.append(test_f1_mi)
                            # 
                            train_acc,train_f1_ma,train_f1_mi = self.acc(train_predict,labels,train_mask)
                            train_accurs.append(train_acc)
                            train_f1_mas.append(train_f1_ma)
                            train_f1_mis.append(train_f1_mi)
                            # 
                
                        # if len(adjs)>2:
                        #     for m in MixGNN_List:
                        #         indexes_to_extract = m.indexes + [-1]  # 
                        #         #  features
                        #         selected_features = [view_features[idx].to(device) for idx in indexes_to_extract]
                        #         fused_embedding = m.fused_graph_learner(torch.cat(selected_features, dim=1))#fused
                        #         learned_fused_adj = m.fused_graph_learner.graph_process(fused_embedding)
                        #         if args.sparse:
                        #             learned_fused_adj.edata['w'] = learned_fused_adj.edata['w'].detach()
                        #         else:
                        #             learned_fused_adj = learned_fused_adj.detach()
                        #         learned_specific_adjs = []
                        #         for i in range(m.mixNum):
                        #             specific_adjs_embedding = m.specific_graph_learner[i](selected_features[i])#specific
                        #             learned_specific_adj = m.specific_graph_learner[i].graph_process(specific_adjs_embedding)
                        #             if args.sparse:
                        #                 learned_specific_adj.edata['w'] = learned_specific_adj.edata['w'].detach()
                        #             else:
                        #                 learned_specific_adj = learned_specific_adj.detach()
                        #             learned_specific_adjs.append(learned_specific_adj)

                        #         predictions = m(features,learned_fused_adj,learned_specific_adjs)
                        #         predictions = [F.softmax(prediction,1) for prediction in predictions]
                        #         predictions = predictions[-1]
                        #         # 
                        #         train_predict =predictions[train_mask]
                        #         test_predict = predictions[test_mask]
                        #         predict_transposed=test_predict.transpose(0, 1)
                        #         matrix.append(predict_transposed)
                        #         train_predict_transposed=train_predict.transpose(0, 1)
                        #         train_matrix.append(train_predict_transposed)
                        #         # 
                        #         test_accur,test_f1_ma,test_f1_mi = self.acc(test_predict,labels,test_mask)
                        #         test_accurs.append(test_accur)
                        #         test_f1_mas.append(test_f1_ma)
                        #         test_f1_mis.append(test_f1_mi)
                        #         # 
                        #         train_acc,train_f1_ma,train_f1_mi = self.acc(train_predict,labels,train_mask)
                        #         train_accurs.append(train_acc)
                        #         train_f1_mas.append(train_f1_ma)
                        #         train_f1_mis.append(train_f1_mi)
                        #         # 
                        #         id = str(m.indexes[0]+1)+'+'+str(m.indexes[1]+1)
                        #         model_id.append(id)


                        matrix_tensor = torch.cat(matrix, dim=0)
                        train_matrix_tensor =torch.cat(train_matrix,dim=0) 
    
                        matrix_tensor_numpy = matrix_tensor.detach().cpu().numpy()
                        train_matrix_numpy = train_matrix_tensor.detach().cpu().numpy()

                        test_accuracy_values = [acc.item() for acc in test_accurs]
                        test_f1_ma_values = [acc.item() for acc in test_f1_mas]
                        test_f1_mi_values = [acc.item() for acc in test_f1_mis]

                        train_accuracy_values = [acc.item() for acc in train_accurs]
                        train_f1_ma_values = [acc.item() for acc in train_f1_mas]
                        train_f1_mi_values = [acc.item() for acc in train_f1_mis]

                        ensemble_acc,_,_ = getCooperationResult(args,train_matrix_numpy,matrix_tensor_numpy,labels[train_mask],labels[test_mask],train_accuracy_values)
                        test_accuracy_values.append(ensemble_acc)
                        
                    return  test_accuracy_values,model, specific_graph_learner,fused_graph_learner,MixGNN_List
import json
def save_args_to_file(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
def add_graphs(learned_adj1, learned_adj2):
    src1, dst1 = learned_adj1.edges()
    w1 = learned_adj1.edata['w']
    
    src2, dst2 = learned_adj2.edges()
    w2 = learned_adj2.edata['w']
    combined_src = torch.cat((src1, src2))
    combined_dst = torch.cat((dst1, dst2))
    combined_weights = torch.cat((w1, w2))
    combined_graph = dgl.graph((combined_src, combined_dst), num_nodes=learned_adj1.num_nodes())

    combined_graph.edata['w'] = combined_weights
    
    return combined_graph
if __name__ == '__main__':
        experiment = Experiment()
        experiment.train(args)
