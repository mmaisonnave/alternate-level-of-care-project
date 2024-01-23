import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
from imblearn.ensemble import BalancedRandomForestClassifier



import numpy as np
from collections import defaultdict
import sys
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import logger

if __name__=='__main__':
    # ---------- ---------- ---------- #
    #         PyTorch                  #
    # ---------- ---------- ---------- #

    # Auxiliary functions   #
    # ---------- ---------- #
    def checkpoint(model, filename):
        torch.save(model.state_dict(), filename)
        
    def resume(model, filename):
        model.load_state_dict(torch.load(filename))

    # Early Stopping        #
    # ---------- ---------- #
    class EarlyStopper():
        def __init__(self, patience, tol):
            self.counter=0
            self.patience = patience
            self.tol = tol
            self.early_stop=False
            self.min_validation_loss = float('inf')
            self.new_min=False
            config_ = configuration.get_config()
            self.logging = logger.init_logger(config_['pytorch_mlp_log'])

        def add_loss(self, validation_loss):
            self.new_min=False
            if validation_loss < self.min_validation_loss:
                self.logging.debug(f'NEW MIN: {validation_loss}, saving weights')
                self.new_min=True
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.tol):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


    # ---------- ---------- ---------- #
    # Train and Test Matrices          #
    # ---------- ---------- ---------- #
    config = configuration.get_config()
    logging = logger.init_logger(config['pytorch_mlp_log'])

    with open(config['experiments_config'], encoding='utf-8') as reader:
        experiment_configurations = json.load(reader)
    configuration_name = 'configuration_13'
    print(f'CONFIG: configuration_name={configuration_name}')
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(experiment_configurations[configuration_name])

    print(f'X_train=      {X_train.shape}')
    print(f'y_train=      {y_train.shape}')
    print(f'X_test=       {X_train.shape}')
    print(f'y_test=       {y_test.shape}')
    print(f'len(columns)= {len(columns)}')


    # pytorch_mlp_config = {
    #     "validation_ratio":0.1,
    #     "hidden_layers": [100, ],
    #     "positive_proportion_importance": 20,
    #     "lr": 0.001,
    #     "patience": 20,
    #     "tol": 1e-4,
    #     "n_epochs": 1000,
    #     "batch_size": 10000,
    #     'early_stopping': True
    # }

    # print('CONFIGURATION USED FOR PyTorch MODELS')
    # for key, value in pytorch_mlp_config.items():
    #     print(f'{key:20}={value}')

    # # ---------- ---------- ---------- ---------- ----------  #
    # # Validation data construction (for early stopping)       #
    # # ---------- ---------- ---------- ---------- ----------  #
    # rng = np.random.default_rng(seed=3076098588645509166)
    # instance_count, feature_count = X_train.shape

    # validation_ratio=pytorch_mlp_config['validation_ratio']


    # print(f'validation_ratio={validation_ratio}')
    # val_ix = set(rng.choice(range(instance_count),
    #                         size=int(validation_ratio*instance_count), 
    #                         replace=False))
    # print(f'len(val_ix)={len(val_ix)}')
    # val_mask = np.array([ix in val_ix for ix in range(instance_count)])

    # X_val_tensor=torch.tensor(X_train.toarray()[val_mask,:], dtype=torch.float32)
    # y_val_tensor=torch.tensor(np.vstack([y_train[val_mask]==0, y_train[val_mask]==1]).T, dtype=torch.float32)

    # X_train_tensor = torch.tensor(X_train.toarray()[~val_mask,:], dtype=torch.float32)
    # y_train_tensor = torch.tensor(np.vstack([y_train[~val_mask]==0, y_train[~val_mask]==1]).T, dtype=torch.float32)


    # X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    # y_test_tensor = torch.tensor(np.vstack([y_test==0, y_test==1]).T, dtype=torch.float32)

    # print(f'X_train_tensor.shape={X_train_tensor.shape}')
    # print(f'y_train_tensor.shape={y_train_tensor.shape}')
    # print()


    # print(f'X_val_tensor.shape={X_val_tensor.shape}')
    # print(f'y_val_tensor.shape={y_val_tensor.shape}')
    # print()


    # print(f'X_test_tensor.shape={X_test_tensor.shape}')
    # print(f'y_test_tensor.shape={y_test_tensor.shape}')
    # print()

    # components = []
    # hidden_layers = pytorch_mlp_config['hidden_layers']
    # logging.debug(f'hidden_layers={hidden_layers}')
    # last_layer_size = feature_count
    # # For each hidden layer we create a linear layer plus a relu activation
    # for ix, size_layer in enumerate(hidden_layers):
    #     components.append(nn.Linear(last_layer_size, size_layer))
    #     components.append(nn.Tanh())
    #     last_layer_size=size_layer

    # # for the output layer we use a 2 dimension linear + softmax (we use 2D for the to classes)
    # components.append(nn.Linear(last_layer_size, 2))
    # components.append(nn.Softmax(dim=1))

    # # We use the list of components to build the neural network #
    # nn_model = nn.Sequential(*components)

    # print(nn_model)


    # instance_count, feature_count = X_train_tensor.shape

    # positive_proportion_importance = pytorch_mlp_config['positive_proportion_importance']
    #                                     # if ==1 then both classes are equally important 
    #                                     # if ==1 equivalent to class_weight==balanced

    # positive_count = np.sum(y_train_tensor[:,1].numpy())
    # negative_count = np.sum(y_train_tensor[:,0].numpy())
    # assert positive_count + negative_count == instance_count

    # w0 = instance_count/((1+positive_proportion_importance)*negative_count)
    # w1 = instance_count/((1+positive_proportion_importance)*positive_count)
    # class_weight = torch.FloatTensor([w0, positive_proportion_importance*w1])

    # logging.debug(f'positive_proportion_importance={positive_proportion_importance}')
    # logging.debug(f'class_weight={class_weight}')




    # logging.debug(f'weights={class_weight}')
    # loss_fn = nn.BCELoss(weight=class_weight)  # binary cross entropy
    # val_loss_fn= nn.BCELoss(weight=class_weight)  # binary cross entropy
    # optimizer = optim.Adam(nn_model.parameters(), lr=pytorch_mlp_config['lr'])

    # # ---------- ---------- ---------- ---------- #
    # # We create an early stopper object           #
    # # ---------- ---------- ---------- ---------- #
    # early_stopper = EarlyStopper(patience=pytorch_mlp_config['patience'], 
    #                             tol=pytorch_mlp_config['tol']
    #                             )
    # logging.debug(f"patience={pytorch_mlp_config['patience']}")
    # logging.debug(f"tol={pytorch_mlp_config['tol']}")
    # # ---------- #
    # # TRAINING   #
    # # ---------- #
    # n_epochs = pytorch_mlp_config['n_epochs']
    # logging.debug(f'n_epochs={n_epochs}')
    # batch_size=pytorch_mlp_config['batch_size']
    # logging.debug(f'batch_size={batch_size}')
    # epoch=0
    # while epoch < n_epochs and not early_stopper.early_stop:
    #     for i in range(0, instance_count, batch_size):
    #         end_of_batch=min(i+batch_size,instance_count)
    #         Xbatch = X_train_tensor[i:end_of_batch]
    #         y_pred = nn_model(Xbatch)
    #         ybatch = y_train_tensor[i:end_of_batch]
    #         loss = loss_fn(y_pred, ybatch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     # Loss for entire training corpus:
    #     with torch.no_grad():
    #         y_pred = nn_model(X_train_tensor)
    #     loss = loss_fn(y_pred, y_train_tensor)
        
    #     # if 'early_stopping' in pytorch_mlp_config and pytorch_mlp_config['early_stopping']:
    #     # ---------- ---------- #
    #     # Validation loss       #
    #     # ---------- ---------- #
    #     with torch.no_grad():
    #         y_pred_val = nn_model(X_val_tensor)
    #     validation_loss_result = val_loss_fn(y_pred_val, y_val_tensor)
    #     print(f'Finished epoch {epoch:4}, latest loss {loss:5.4f} val loss {validation_loss_result:5.4f} ')

    #     if 'early_stopping' in pytorch_mlp_config and pytorch_mlp_config['early_stopping']:
    #         # logging.debug('USING VALIDATION LOSS FOR EARLY STOPPING')

    #         early_stopper.add_loss(validation_loss_result.numpy())
    #     else:
    #         # logging.debug('USING TRAINING LOSS FOR EARLY STOPPING')
    #         early_stopper.add_loss(loss.numpy())
    #     if early_stopper.new_min:
    #         # ---------- ---------- ---------- ---------- ---------- ---------- #
    #         # We save the latest checkpoint with the lowest validation loss     #
    #         # ---------- ---------- ---------- ---------- ---------- ---------- #
    #         checkpoint(nn_model, config['pytorch_temp_checkpoint'])
    #     epoch+=1

    #     if early_stopper.early_stop:
    #         logging.debug('Exiting due to Early stopping, recovering best model ...')
    #         resume(nn_model, config['pytorch_temp_checkpoint'])


    # # ---------- ---------- ---------- ---------- ---------- ---------- #
    # # TRAINING FINISHED, COMPUTING METRICS (TRAINING SPLIT)             #
    # # ---------- ---------- ---------- ---------- ---------- ---------- #
    # with torch.no_grad():
    #     y_score = nn_model(X_train_tensor)[:,1]
    # # ---------- ---------- ---------- ---------- ---------- #
    # # Predictions (binary) and scores (real number)          #
    # # ---------- ---------- ---------- ---------- ---------- #
    # y_pred = y_score.round().numpy()
    # y_score = y_score.numpy()
    # y_true = y_train_tensor[:,1].numpy()

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # training_results = {'Model Name': 'PyTorch MLP',
    #                     'Split': 'TRAIN',
    #                     'Precision': precision_score(y_true, y_pred),
    #                     'Recal': recall_score(y_true, y_pred),
    #                     'F1-Score': f1_score(y_true, y_pred),
    #                     'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
    #                     'TN': tn,
    #                     'TP': tp,
    #                     'FN': fn,
    #                     'FP': fp,
    #                     }

    # # ---------- ---------- ---------- ---------- #
    # # COMPUTING METRICS (TESTING SPLIT)           #
    # # ---------- ---------- ---------- ---------- #
    # with torch.no_grad():
    #     y_score = nn_model(X_test_tensor)[:,1]
        
    # y_pred = y_score.round().numpy()
    # y_score = y_score.numpy()
    # y_true = y_test_tensor[:,1].numpy()

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # testing_results = {'Model Name': 'PyTorch MLP',
    #                 'Split': 'TEST',
    #                 'Precision': precision_score(y_true, y_pred),
    #                 'Recal': recall_score(y_true, y_pred),
    #                 'F1-Score': f1_score(y_true, y_pred),
    #                 'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
    #                 'TN': tn,
    #                 'TP': tp,
    #                 'FN': fn,
    #                 'FP': fp
    #                 }

    # # ---------- ---------- ---------- ---------- #
    # # APPENDING NEW RESULTS TO RESULT FILES       #
    # # ---------- ---------- ---------- ---------- #
    # results = {key: [training_results[key], testing_results[key]] for key in training_results.keys()}

    # for key,value in pytorch_mlp_config.items():
    #     print(f"{key:10}: {value}")

    # df = pd.DataFrame(results)

    # for row_number in range(df.shape[0]):
    #     for column_number in range(df.shape[1]):
    #         column_name = df.columns[column_number]
    #         print(f'{column_name:20}={df.iloc[row_number, column_number]}')
    #     print()



    # model = BalancedRandomForestClassifier(n_estimators=500, 
    #                                     sampling_strategy='majority', 
    #                                     replacement=True, 
    #                                     class_weight='balanced_subsample')


    with open(config['models_config'], encoding='utf-8') as reader:
        model_configurations = json.load(reader)
    MODEL_CONFIGURATION_NAME = 'model_104'
    print(f'MODEL CONFIG: model_configuration_name={MODEL_CONFIGURATION_NAME}')
    print()
    MODEL_SEED = 1270833263
    model_random_state=np.random.RandomState(MODEL_SEED)
    model = configuration.model_from_configuration(model_configurations[MODEL_CONFIGURATION_NAME], 
                                                   random_state=model_random_state)
    print(str(model))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    y_score = model.predict_proba(X_train)[:,1]
    y_true = y_train


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    training_results = {'Model Name': 'BRF',
                        'Split': 'TRAIN',
                        'Precision': precision_score(y_true, y_pred),
                        'Recal': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        }

    # ---------- ---------- ---------- ---------- #
    # COMPUTING METRICS (TESTING SPLIT)           #
    # ---------- ---------- ---------- ---------- #
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    y_true = y_test

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    testing_results = {'Model Name': 'BRF',
                       'Split': 'TEST',
                       'Precision': precision_score(y_true, y_pred),
                       'Recal': recall_score(y_true, y_pred),
                       'F1-Score': f1_score(y_true, y_pred),
                       'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                       'TN': tn,
                       'TP': tp,
                       'FN': fn,
                       'FP': fp
                       }

    # ---------- ---------- ---------- ---------- #
    # APPENDING NEW RESULTS TO RESULT FILES       #
    # ---------- ---------- ---------- ---------- #
    results = {key: [training_results[key], testing_results[key]] for key in training_results.keys()}

    df = pd.DataFrame(results)

    for row_number in range(df.shape[0]):
        for column_number in range(df.shape[1]):
            column_name = df.columns[column_number]
            print(f'{column_name:20}={df.iloc[row_number, column_number]}')
        print()