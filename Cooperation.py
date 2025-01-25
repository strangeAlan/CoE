# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,adjusted_rand_score, normalized_mutual_info_score 
# from xgboost import XGBClassifier
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
import pandas as pd 

def Train(X, y, k,deep=10):
   
    c = len(np.unique(y))  
    
    classifiers = []
    G = np.zeros((k * c, len(X)))  
    accuracies = []
    
    for i in range(k):
        size = round(len(X)*0.95)
        bootstrap_indices = np.random.choice(len(X), size=size, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        classifier = DecisionTreeClassifier(max_depth=deep,random_state=1)#LogisticRegression()#classifier = DecisionTreeClassifier(max_depth=4)# # 

        classifier.fit(X_bootstrap, y_bootstrap)
        classifiers.append(classifier)
        
        y_pred = classifier.predict(X) - 1
        
        G[i*c:i*c+c,:]=classifier.predict_proba(X).T
        

        accuracy = classifier.score(X, y)
        accuracies.append(accuracy)
    
    return classifiers, G, accuracies


def PredictG(classifiers, X,c):
    k = len(classifiers)
    n = len(X)
    
    G = np.zeros((k * c, n))  
    

    for i, clf in enumerate(classifiers):
        y_pred = clf.predict(X)-1
        G[i*c:i*c+c,:]=clf.predict_proba(X).T
    
    return G



def OneHot(y):
    y=y.astype(int)

    n = len(y)  
    c = int(np.max(y))  
    
    Y = np.zeros((c, n))  
    
    for i in range(n):
        Y[y[i]-1 , i] = 1  
    
    return Y


def init_Theta(accuracies,c):
    replicated = np.repeat(accuracies, c)
    stacked = np.tile(replicated, (c, 1))
    
    return stacked



def Loss(Theta,G,Y,gamma,alpha=100):# 100 default
    #Theta=th.tensor(Theta,dtype=th.float,requires_grad=True)
    G=th.tensor(G,dtype=th.float)
    Y=th.tensor(Y,dtype=th.float)

    n=G.shape[1]
    ThetaG=th.mm(Theta,G)
    Prob=th.softmax(ThetaG, dim=0)
    L1= th.sum(-Y*th.log(Prob))
    #YProb=th.diag(th.mm(Y.T,Prob))
    YProb=Y*Prob
    MProb=Y*Prob
    #MProb=Y*th.max(Prob,dim=0)[0]
    Max_2=th.logsumexp(alpha * (Prob-MProb), dim=0)/alpha
    L2=-gamma*th.sum(th.sum(YProb,dim=0)-Max_2)
    #YThetaG=th.diag(th.mm(Y.T,ThetaG))
    #Max_2=th.logsumexp(alpha * (ThetaG-YThetaG), dim=0)/alpha
    #L2=-gamma*th.sum(YThetaG-Max_2)
    L=(L1+L2)/n#+2*th.norm(Theta,1))/n
    # print(L1,L2)
    return L


def generate_ring_data(num_samples, inner_radius, outer_radius, noise_std):
    theta_outer = np.random.rand(num_samples) * 2 * np.pi
    x_outer = outer_radius * np.cos(theta_outer)+0.2*noise_std * np.random.randn(num_samples)
    y_outer = outer_radius * np.sin(theta_outer)+0.2*noise_std * np.random.randn(num_samples)

    theta_inner = np.random.rand(num_samples) * 2 * np.pi
    x_inner = (inner_radius + noise_std * np.random.randn(num_samples)) * np.cos(theta_inner)
    y_inner = (inner_radius + noise_std * np.random.randn(num_samples)) * np.sin(theta_inner)

    x = np.concatenate((x_outer, x_inner))
    y = np.concatenate((y_outer, y_inner))

    labels = np.ones(num_samples * 2)
    labels[num_samples:] = 2

    return x, y, labels



        
def getCooperationResult(args,G,G_test,y_train,y_test,accuracies):
    y_train = y_train.cpu().numpy() + 1
    y_test = y_test.cpu().numpy()+1
    gamma=100
    loss_array=np.ones(3000)
    Y=OneHot(y_train)
    
    c= len(np.unique(y_train))
    Theta=init_Theta(accuracies, c)
    Theta=th.tensor(Theta,requires_grad=True,dtype=th.float)
    G = th.tensor(G)
    G_test = th.tensor(G_test)
    optimizer = optim.Adam([Theta], lr=.002)

    for epoch in range(args.ensemble_epoch):
        optimizer.zero_grad()  
        loss = Loss(Theta, G, Y, gamma)  
        loss_array[epoch]=loss.detach().numpy()
        loss.backward()  
        optimizer.step()  
    
        # if epoch % args.ensemble_eval_freq == 0:
        #     print(f'Epoch [{epoch+1}/n], Loss: {loss.item()}')
    # 
    # optimal_Theta = Theta.detach().numpy()
    y_coe_test= th.argmax(th.mm(Theta,G_test), dim=0).numpy()+1
    f1_coe_test_ma = f1_score(y_coe_test,y_test,average='macro')
    f1_coe_test_mi = f1_score(y_coe_test,y_test,average='micro')
    acc_coe_test = accuracy_score(y_coe_test,y_test)
    return acc_coe_test,f1_coe_test_ma,f1_coe_test_mi
