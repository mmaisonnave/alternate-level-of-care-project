import numpy as np
import pandas as pd

import os

import sys

sys.path.append('..')

from utilities import health_data
from utilities import configuration


if __name__ == '__main__':
    FEATURES=[
              
              # Group I
              'acute_days',
              'cmg',
              'New Acute Patient',
              'Unplanned Readmit',

              # Group II
              'alc_days',
              'Day Surgery Entry',
              'Emergency Entry',
              'General Surgery'
              'level 1 comorbidity',
              'transfusion given',
              'urgent admission',

              # Group III
              'age',
              'case_weight',
              'Direct Entry',
              'elective admission',
              'Family Practice',
              'female',
              'General Medicine',
              'is alc',
              'is central zone',
              'level 4 comorbidity',
              'male',
              'OBS Delivered',
              'Oral Surgery',
              'Orthopaedic Surgery',
              'Palliative Care',
              'Panned Readmit',
              'Psychiatry',
              'Urology',

              # Group I D+i
              'j441',
              'i500',
              'z515',
              'z38000',
              '5md50aa'
              ]
    params = configuration.configuration_from_configuration_name('configuration_93')


    X_train, y_train, X_test, y_test, columns =  health_data.Admission.get_train_test_matrices(params)
    
    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')

    print(f'X_test.shape={X_test.shape}')
    print(f'y_test.shape={y_test.shape}')

    columns = list(columns)
    indexes = [ix for ix in [columns.index(feature) for feature in FEATURES]]

    
    X_train = X_train[:,indexes].toarray()
    X_test = X_test[:,indexes].toarray()

    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')

    print(f'X_test.shape={X_test.shape}')
    print(f'y_test.shape={y_test.shape}')


    train_df = pd.DataFrame(np.hstack([X_train, y_train.reshape(-1,1)]),
                            columns=FEATURES+['target'])
    
    test_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1,1)]),
                           columns=FEATURES+['target'])
    
    train_df.to_csv('train_df.csv', index=None)
    test_df.to_csv('test_df.csv', index=None)
    