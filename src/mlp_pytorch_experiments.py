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

    def add_validation_loss(self, validation_loss):
        self.new_min=False
        if validation_loss < self.min_validation_loss:
            print(f'NEW MIN: {validation_loss}, saving weights')
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


    for experiment_configuration_name in [experiment_configuration_name for _,experiment_configuration_name in to_do]:
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
                # Randomly selecting instances to be part of the validation set:
                val_ix = set(rng.choice(range(instance_count),
                                        size=int(validation_ratio*instance_count), 
                                        replace=False))

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

                # ---------- ---------- ---------- ---------- # 
                # Building Neural Network Architecture        #
                # ---------- ---------- ---------- ---------- #
                components = []
                hidden_layers = pytorch_mlp_config['hidden_layers']
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
                print(model)

                # ---------- ---------- ---------- #
                # Loss function and optimizer      #
                # ---------- ---------- ---------- #
                weights = torch.FloatTensor(pytorch_mlp_config['class_weight']) 
                loss_fn = nn.BCELoss(weight=weights)  # binary cross entropy
                val_loss_fn= nn.BCELoss(weight=weights)  # binary cross entropy
                optimizer = optim.Adam(model.parameters(), lr=pytorch_mlp_config['lr'])

                # ---------- ---------- ---------- ---------- #
                # We create an early stopper object           #
                # ---------- ---------- ---------- ---------- #
                early_stopper = EarlyStopper(patience=pytorch_mlp_config['patience'], 
                                            tol=pytorch_mlp_config['tol']
                                            )
                # ---------- #
                # TRAINING   #
                # ---------- #
                n_epochs = pytorch_mlp_config['n_epochs']
                batch_size=pytorch_mlp_config['batch_size']
                epoch=0
                while epoch < n_epochs and not early_stopper.early_stop:
                    for i in range(0, instance_count, batch_size):
                        Xbatch = X_train_tensor[i:i+batch_size]
                        y_pred = model(Xbatch)
                        ybatch = y_train_tensor[i:i+batch_size]
                        loss = loss_fn(y_pred, ybatch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    print(f'Loss: {loss:4.3f}')
                    # ---------- ---------- #
                    # Validation loss       #
                    # ---------- ---------- #
                    with torch.no_grad():
                        y_pred_val = model(X_val_tensor)
                    validation_loss_result = val_loss_fn(y_pred_val, y_val_tensor)

                    early_stopper.add_validation_loss(validation_loss_result.numpy())
                    if early_stopper.new_min:
                        # ---------- ---------- ---------- ---------- ---------- ---------- #
                        # We save the latest checkpoint with the lowest validation loss     #
                        # ---------- ---------- ---------- ---------- ---------- ---------- #
                        checkpoint(model, config['pytorch_temp_checkpoint'])
                    epoch+=1
                    print(f'Finished epoch {epoch:4}, latest loss {loss:5.4f} val loss {validation_loss_result:5.4f} ')

                    if early_stopper.early_stop:
                        print('Exiting due to Early stopping, recovering best model ...')
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
