import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import json

import sys 
sys.path.append('..')

from utilities import health_data
from utilities import configuration


if __name__ == '__main__':
    config = configuration.get_config()

    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(experiment_configurations['configuration_15'])

    print(f'X_train=      {X_train.shape}')
    print(f'y_train=      {y_train.shape}')
    print(f'X_test=       {X_train.shape}')
    print(f'y_test=       {y_test.shape}')
    print(f'len(columns)= {len(columns)}')

    X = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y = torch.tensor(np.vstack([y_train==0, y_train==1]).T, dtype=torch.float32)


    feature_count=X.shape[1]
    components = []
    hidden_layers = [100,]
    last_layer_size = feature_count


    for ix, size_layer in enumerate(hidden_layers):
        components.append(nn.Linear(last_layer_size, size_layer))
        components.append(nn.ReLU())
        last_layer_size=size_layer


    components.append(nn.Linear(last_layer_size, 2))

    components.append(nn.Softmax(dim=1))

    model = nn.Sequential(*components)
    print(model)

    weights = torch.FloatTensor([0.5, 12]) 
    loss_fn = nn.BCELoss(weight=weights)  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 200
    batch_size=100

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')


    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_score = model(X)[:,1]
        
    y_pred = y_score.round().numpy()
    y_score = y_score.numpy()
    y_true = y[:,1].numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f'precision={precision_score(y_true, y_pred):5.4f}')
    print(f'recall=   {recall_score(y_true, y_pred):5.4f}')
    print(f'f1_score= {f1_score(y_true, y_pred):5.4f}')
    print()
    print(f'TP={tp}')
    print(f'TN={tn}')
    print(f'FP={fp}')
    print(f'FN={fn}')
    print()
    print(f'AUC= {roc_auc_score(y_true=y_true, y_score=y_score):5.4f}')