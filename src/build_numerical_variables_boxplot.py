import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from utilities import health_data 
from utilities import configuration

if __name__ == "__main__":
    config = configuration.get_config()

    CONFIGURATION_ID='configuration_27' # Only numerical with missing fixed.
    print(f'Running on configuration ID: {CONFIGURATION_ID}')
    params = configuration.configuration_from_configuration_name(CONFIGURATION_ID)
        
    # Computing training and testing matrices.
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(params)
    print(f'X_train.shape = {X_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'X_train.shape = {X_test.shape}')
    print(f'y_train.shape = {y_test.shape}')

    train_and_test_X = np.vstack([X_train.toarray(),
                                  X_test.toarray()])

    _, variable_count = X_train.shape

    for i in range(variable_count):
        fig, ax = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(10,10))
        serie = train_and_test_X[:,i]
        if feature_names[i]=='acute_days' or feature_names[i]=='alc_days':
            print(f'Discarding={np.sum(serie<=0):5,} elements for feature={feature_names[i]}')
            print(f'Using=     {np.sum(serie>0):5,} elements for feature={feature_names[i]}')
            serie = serie[serie>0]
        ax.hist(serie, bins=50)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        ax.set_ylabel(feature_names[i])
        if feature_names[i]=='case_weight' or feature_names[i]=='acute_days' or feature_names[i]=='alc_days':
            ax.set_xscale('log')
            ax.set_yscale('log')

        fig.savefig(config['numerical_variables_plot'][:-4]+f'_{feature_names[i]}.jpg', bbox_inches='tight')

    # for i in [2, 3, 4]:
    #     fig, ax = plt.subplots(nrows=1,
    #                         ncols=1,
    #                         figsize=(10,10))
    #     ax.hist(train_and_test_X[:,i], bins=50)
    #     ax.set_xscale('log')
    #     # ax.set_yscale('log')

    #     ax.tick_params(axis='both', which='major', labelsize=11)
    #     ax.tick_params(axis='both', which='minor', labelsize=11)
    #     ax.set_ylabel(feature_names[i])
    #     fig.savefig(config['numerical_variables_plot'][:-4]+f'_{i}.jpg', bbox_inches='tight')
    print('Done!')
