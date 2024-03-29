"""
This modules contains a main function with two nested procedures (checkpoint and resume) and 
one nested class (EarlyStopper).

The main function runs all the MultiLayerPerceptron (Feed forward neural network) experiments using 
the PyTorch library. 

The main function takes the following inputs:
- 'pytorch_mlp.json': A configuration file describing the architecture and hyerparameters of 
                      all MLP models
- experiment_configuration.json: A configuration file describing the preprocessings steps 
                                 to build the training and testing matrices.

For example the first PyTorch MLP model is a single 100-dimension hidden layer architecture:
    "pytorch_conf1": {
        "description": "Base model",
        "validation_ratio":0.25,
        "hidden_layers": [100],
        "positive_proportion_importance": 1,
        "lr": 0.001,
        "patience": 50,
        "tol": 1e-4,
        "n_epochs": 500,
        "batch_size": 512,
        "configuration_ids": ["configuration_15"]
    },

This first model (pytorch_conf1) runs using configuration_15:
    "configuration_15": {
            "fix_skew": false,
            "normalize": false,
            "fix_missing_in_testing": true,
            "numerical_features": true,
            "categorical_features": true,
            "diagnosis_features": false,
            "intervention_features": false,
            "use_idf": false,
            "class_balanced": false,
            "remove_outliers": true,
            "under_sample_majority_class": false,
            "over_sample_minority_class": false,
            "smote_and_undersampling":false,
            "diagnosis_embeddings": true,
            "intervention_embeddings": true,
            "diag_embedding_model_name": "diag_conf_1",
            "interv_embedding_model_name": "interv_conf_1"
        },
This configuration uses the latest features, including diagnosis and intervention embeddings. 

The results of the models are stored in the output file: 'pytorch_mlp.csv' 
(path obtained from paths.yaml)

The overview of the main function is as follows:
1. Define auxiliary procedures and nested class
2. Computing pending experiments based on the config files
3. Computing how many experiments are already ran based on results stored in 'pytorch_mlp.csv' file
4. for experiment_config in experiment_configuration.json':
5.     X_train, y_train, X_test, y_test <= get_matrices_from config(experiment_config)
6.     for pytorch_config in 'pytorch_mlp.json':
7.         split train into train and validation (to perform early stopping using the val loss)
8.         calculating balanced loss weights 
9.         creating model using the number of hidden layers specified in pytorch_config
10.        training Neural Network using Early stopping.
11.        compute_training_metrics(model, X_train, y_train)
12.        compute_testing_metrics(model, X_test, y_test)
13.        append new results to existing result file ('pytorch_mlp.csv')

"""
import numpy as np

import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import json

import sys 
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import logger

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

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


if __name__ == '__main__':
    # ---------- ---------- ---------- ---------- #
    # RETRIEVING CONFIGURATION FILES AND LOGGER   # 
    # ---------- ---------- ---------- ---------- #
    config = configuration.get_config()
    logging = logger.init_logger(config['pytorch_mlp_log'])

    with open(config['pytorch_mlp_model_configurations'], encoding='utf-8') as reader:
        pytorch_mlp_configs = json.load(reader)
    with open(config['experiments_config'], encoding='utf-8') as reader:
        experiment_configurations = json.load(reader)

    logging.debug(f"Running pytorch conf from: {config['pytorch_mlp_model_configurations']}")
    logging.debug(f"Running experim conf from: {config['experiments_config']}")

    # # keeping only configurations used for pytorch_mlp models.
    # used_configs = set([experiment_configuration_name for _, ptmlp_config in pytorch_mlp_configs.items() 
    #                    for experiment_configuration_name in ptmlp_config['configuration_ids']])
    

    # ---------- ---------- ---------- ---------- ---------- #
    # CALCULATING NUMBER OF MODELS AND CONFIGURATIONS TO RUN #
    # ---------- ---------- ---------- ---------- ---------- #
    to_do=[]
    for experiment_configuration_name in experiment_configurations.keys():
        for pytorch_mlp_config_name, pytorch_mlp_config in pytorch_mlp_configs.items():
            if experiment_configuration_name in pytorch_mlp_config['configuration_ids']:
                # For each pair of model and experiment configuration append to to-do list.
                to_do.append((pytorch_mlp_config_name, experiment_configuration_name))
    
    logging.debug(f'Found {len(to_do)} experiments to run (some might be skipped if already ran).')

    # ---------- ---------- ---------- ---------- ---------- ---------- #
    # Removing from the to-do list those experiments already ran        #
    # ---------- ---------- ---------- ---------- ---------- ---------- #
    if os.path.isfile(config['pytorch_mlp_results']):
        old_df = pd.read_csv(config['pytorch_mlp_results'], sep=';')
        count_already_run=0
        for model_config_name, experiment_config_name in zip(old_df['pytorch_config_name'], 
                                                             old_df['experiment_config_name']):
            if (model_config_name, experiment_config_name) in to_do:
                to_do.remove((model_config_name, experiment_config_name))
                count_already_run+=1
        logging.debug(f'Found {count_already_run} experiments already ran.')
    logging.debug(f'Running {len(to_do)} experiments....')


    for experiment_configuration_name in set([experiment_configuration_name for _,experiment_configuration_name in to_do]):
        experiment_configuration = experiment_configurations[experiment_configuration_name]
        logging.debug(f'Working with the experiment configuration: {experiment_configuration_name}')
        X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(experiment_configuration)

        logging.debug(f'X_train=      {X_train.shape}')
        logging.debug(f'y_train=      {y_train.shape}')
        logging.debug(f'X_test=       {X_train.shape}')
        logging.debug(f'y_test=       {y_test.shape}')
        logging.debug(f'len(columns)= {len(columns)}')

        for pytorch_mlp_config_name in [pytorch_mlp_config_name for pytorch_mlp_config_name,_ in to_do]:
            pytorch_mlp_config = pytorch_mlp_configs[pytorch_mlp_config_name]
            logging.debug(f'Working the model configuration: {pytorch_mlp_config_name}')

            # Only running the configuration pair (model,experiment) if the current 
            # model (pytorch_mlp_config_name) was set up
            # to run with the current configuration(experiment_configuration_name)
            if experiment_configuration_name in pytorch_mlp_config['configuration_ids']:
                logging.debug(f'RUNNING: {(pytorch_mlp_config_name, experiment_configuration_name)}')
                # ---------- ---------- ---------- ---------- ----------  #
                # Validation data construction (for early stopping)       #
                # ---------- ---------- ---------- ---------- ----------  #
                rng = np.random.default_rng(seed=3076098588645509166)
                instance_count, feature_count = X_train.shape

                validation_ratio=pytorch_mlp_config['validation_ratio']
                logging.debug(f'validation_ratio={validation_ratio}')
                # Randomly selecting instances to be part of the validation set:
                val_ix = set(rng.choice(range(instance_count),
                                        size=int(validation_ratio*instance_count), 
                                        replace=False))
                logging.debug(f'len(val_ix)={len(val_ix)}')
                val_mask = np.array([ix in val_ix for ix in range(instance_count)])

                # Creating validation data and transforming everything into PyTorch tensors.
                X_val_tensor=torch.tensor(X_train.toarray()[val_mask,:], dtype=torch.float32)
                y_val_tensor=torch.tensor(np.vstack([y_train[val_mask]==0, y_train[val_mask]==1]).T, dtype=torch.float32)

                X_train_tensor = torch.tensor(X_train.toarray()[~val_mask,:], dtype=torch.float32)
                y_train_tensor = torch.tensor(np.vstack([y_train[~val_mask]==0, y_train[~val_mask]==1]).T, dtype=torch.float32)


                X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
                y_test_tensor = torch.tensor(np.vstack([y_test==0, y_test==1]).T, dtype=torch.float32)



                logging.debug(f'X_train_tensor.shape=    {X_train_tensor.shape}')
                logging.debug(f'y_train_tensor.shape=    {y_train_tensor.shape}')

                logging.debug(f'X_val_tensor.shape=      {X_val_tensor.shape}')
                logging.debug(f'y_val_tensor.shape=      {y_val_tensor.shape}')

                logging.debug(f'X_test_tensor.shape=     {X_test_tensor.shape}')
                logging.debug(f'y_test_tensor.shape=     {y_test_tensor.shape}')

                # new instance count after computing validation split
                instance_count, feature_count = X_train_tensor.shape
                logging.debug(f'instance_count={instance_count}')
                logging.debug(f'feature_count={feature_count}')
                logging.debug(f'Number of positives in training={np.sum(y_train_tensor[:,1].numpy())}')

                # ---------- ---------- ---------- ---------- ---------- ---------- 
                positive_count = np.sum(y_train_tensor[:,1].numpy())
                negative_count = np.sum(y_train_tensor[:,0].numpy())
                assert positive_count + negative_count == instance_count
                positive_proportion_importance = pytorch_mlp_config['positive_proportion_importance']
                                                    # if ==1 then both classes are equally important 
                                                    # if ==1 equivalent to class_weight==balanced
                w0 = instance_count/((1+positive_proportion_importance)*negative_count)
                w1 = instance_count/((1+positive_proportion_importance)*positive_count)
                class_weight = torch.FloatTensor([w0, positive_proportion_importance*w1])

                logging.debug(f'positive_proportion_importance={positive_proportion_importance}')
                logging.debug(f'class_weight={class_weight}')
                # ---------- ---------- ---------- ---------- ---------- ---------- 



                # ---------- ---------- ---------- ---------- # 
                # Building Neural Network Architecture        #
                # ---------- ---------- ---------- ---------- #
                components = []
                hidden_layers = pytorch_mlp_config['hidden_layers']
                logging.debug(f'hidden_layers={hidden_layers}')
                last_layer_size = feature_count
                # For each hidden layer we create a linear layer plus a relu activation
                for ix, size_layer in enumerate(hidden_layers):
                    components.append(nn.Linear(last_layer_size, size_layer))
                    components.append(nn.ReLU())
                    last_layer_size=size_layer

                # for the output layer we use a 2 dimension linear + softmax (we use 2D for the to classes)
                components.append(nn.Linear(last_layer_size, 2))
                components.append(nn.Softmax(dim=1))

                # We use the list of components to build the neural network #
                model = nn.Sequential(*components)
                logging.debug(model)

                # ---------- ---------- ---------- #
                # Loss function and optimizer      #
                # ---------- ---------- ---------- #
                # weights = torch.FloatTensor(pytorch_mlp_config['class_weight']) 
                # weights = torch.FloatTensor(pytorch_mlp_config['class_weight']) 
                logging.debug(f'weights={class_weight}')
                loss_fn = nn.BCELoss(weight=class_weight)  # binary cross entropy
                val_loss_fn= nn.BCELoss(weight=class_weight)  # binary cross entropy
                optimizer = optim.Adam(model.parameters(), lr=pytorch_mlp_config['lr'])

                # ---------- ---------- ---------- ---------- #
                # We create an early stopper object           #
                # ---------- ---------- ---------- ---------- #
                early_stopper = EarlyStopper(patience=pytorch_mlp_config['patience'],
                                            tol=pytorch_mlp_config['tol']
                                            )
                logging.debug(f"patience={pytorch_mlp_config['patience']}")
                logging.debug(f"tol={pytorch_mlp_config['tol']}")
                # ---------- #
                # TRAINING   #
                # ---------- #
                n_epochs = pytorch_mlp_config['n_epochs']
                logging.debug(f'n_epochs={n_epochs}')
                batch_size=pytorch_mlp_config['batch_size']
                logging.debug(f'batch_size={batch_size}')
                epoch=0
                while epoch < n_epochs and not early_stopper.early_stop:
                    for i in range(0, instance_count, batch_size):
                        end_of_batch=min(i+batch_size,instance_count)
                        Xbatch = X_train_tensor[i:end_of_batch]
                        y_pred = model(Xbatch)
                        ybatch = y_train_tensor[i:end_of_batch]
                        loss = loss_fn(y_pred, ybatch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Loss for entire training corpus:
                    with torch.no_grad():
                        y_pred = model(X_train_tensor)
                    loss = loss_fn(y_pred, y_train_tensor)
                    # ---------- ---------- #
                    # Validation loss       #
                    # ---------- ---------- #
                    with torch.no_grad():
                        y_pred_val = model(X_val_tensor)
                    validation_loss_result = val_loss_fn(y_pred_val, y_val_tensor)
                    logging.debug(f'Finished epoch {epoch:4}, latest loss {loss:5.4f} val loss {validation_loss_result:5.4f} ')

                    if 'early_stopping' in pytorch_mlp_config and pytorch_mlp_config['early_stopping']:
                        logging.debug('USING VALIDATION LOSS FOR EARLY STOPPING')

                        early_stopper.add_loss(validation_loss_result.numpy())
                    else:
                        logging.debug('USING TRAINING LOSS FOR EARLY STOPPING')
                        early_stopper.add_loss(loss.numpy())
                    if early_stopper.new_min:
                        # ---------- ---------- ---------- ---------- ---------- ---------- #
                        # We save the latest checkpoint with the lowest validation loss     #
                        # ---------- ---------- ---------- ---------- ---------- ---------- #
                        checkpoint(model, config['pytorch_temp_checkpoint'])
                    epoch+=1

                    if early_stopper.early_stop:
                        logging.debug('Exiting due to Early stopping, recovering best model ...')
                        resume(model, config['pytorch_temp_checkpoint'])
                    
                    
                # ---------- ---------- ---------- ---------- ---------- ---------- #
                # TRAINING FINISHED, COMPUTING METRICS (TRAINING SPLIT)             #
                # ---------- ---------- ---------- ---------- ---------- ---------- #
                with torch.no_grad():
                    y_score = model(X_train_tensor)[:,1]
                
                # ---------- ---------- ---------- ---------- ---------- #
                # Predictions (binary) and scores (real number)          #
                # ---------- ---------- ---------- ---------- ---------- #
                y_pred = y_score.round().numpy()
                y_score = y_score.numpy()
                y_true = y_train_tensor[:,1].numpy()

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                training_results = {'Split': 'TRAIN',
                                    'Precision': precision_score(y_true, y_pred),
                                    'Recal': recall_score(y_true, y_pred),
                                    'F1-Score': f1_score(y_true, y_pred),
                                    'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                                    'TN': tn,
                                    'TP': tp,
                                    'FN': fn,
                                    'FP': fp,
                                    'pytorch_config_name': pytorch_mlp_config_name,
                                    'experiment_config_name': experiment_configuration_name
                                    }

                # ---------- ---------- ---------- ---------- #
                # COMPUTING METRICS (TESTING SPLIT)           #
                # ---------- ---------- ---------- ---------- #
                with torch.no_grad():
                    y_score = model(X_test_tensor)[:,1]
                    
                y_pred = y_score.round().numpy()
                y_score = y_score.numpy()
                y_true = y_test_tensor[:,1].numpy()

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                testing_results = {'Split': 'TEST',
                                'Precision': precision_score(y_true, y_pred),
                                'Recal': recall_score(y_true, y_pred),
                                'F1-Score': f1_score(y_true, y_pred),
                                'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                                'TN': tn,
                                'TP': tp,
                                'FN': fn,
                                'FP': fp,
                                'pytorch_config_name': pytorch_mlp_config_name,
                                'experiment_config_name': experiment_configuration_name
                                }
                
                # ---------- ---------- ---------- ---------- #
                # APPENDING NEW RESULTS TO RESULT FILES       #
                # ---------- ---------- ---------- ---------- #
                results = {key: [training_results[key], testing_results[key]] for key in training_results.keys()}

                results_df = pd.DataFrame(results)
                if os.path.isfile(config['pytorch_mlp_results']):
                    old_df = pd.read_csv(config['pytorch_mlp_results'], sep=';')
                    results_df = pd.concat([old_df,results_df])

                results_df.to_csv(config['pytorch_mlp_results'], index=False, sep=';')
