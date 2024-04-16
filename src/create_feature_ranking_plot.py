import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

import sys
sys.path.append('..')

from utilities import configuration

def _capitalize_feature_name(feature_name:str)->str:
    if feature_name=='cmg':
        return 'CMG'
    elif feature_name=='case_weight':
        return 'RIW'
    else:
        aux = feature_name.replace('_', ' ').replace('-', ' ').strip()
        if aux.split(' ')=='':
            print('Error')
            print(aux)
        return ' '.join([word[0].upper()+word[1:] for word in aux.split(' ')])

if __name__ == "__main__":
    FIGSIZE=(5, 12)
    FILTER_NUMERIC = True

    # sns.set_style('whitegrid')
    config = configuration.get_config()

    # Permutation Feature Importance Results
    input_filename = config['permutation_feature_importance_results']
    input_filename = input_filename[:-4] + "_only_num_and_cat.csv"
    pfi_results_df = pd.read_csv(input_filename)
    assert list(pfi_results_df.columns)==['variable', 'importances_mean', 'importances_std']
    # permutation_feature_importance_results['Ranking'] = [ix+1 for ix in range(permutation_feature_importance_results.shape[0])]

    # Logisitc Regression Coefficients
    LR_coef_df = pd.read_csv(config['explainable_logreg_coefficients'])
    assert list(LR_coef_df.columns) == ['Feature Name', 'Score', 'Code Description']
    # Sorting with absolute score to LogReg
    LR_coef_df['Absolute Score'] = np.abs(LR_coef_df['Score'])
    LR_coef_df = LR_coef_df.sort_values(by=['Absolute Score'], ascending=False)

    ordered_vars_LR = list(LR_coef_df['Feature Name'])
    ordered_vars_PFI = list(pfi_results_df['variable'])

    print('Before capitalizing')
    print(ordered_vars_LR)

    ordered_vars_LR=list(map(_capitalize_feature_name, ordered_vars_LR))
    ordered_vars_PFI=list(map(_capitalize_feature_name, ordered_vars_PFI))

    print('AFter capitalizing')
    print(ordered_vars_LR)


    all_features = set(ordered_vars_LR).intersection(ordered_vars_PFI)
    all_features = np.array(list(all_features))
    feature_count = all_features.shape[0]
    print(f'feature_count={feature_count}')

    print(f'len(ordered_vars_LR)={len(ordered_vars_LR)}')
    print(f'len(ordered_vars_PFI)={len(ordered_vars_PFI)}')
    print('Filtering ...')

    #filtering 
    ordered_vars_LR = list(filter(lambda feature: feature in all_features, ordered_vars_LR))
    ordered_vars_PFI = list(filter(lambda feature: feature in all_features, ordered_vars_PFI))

    print(f'len(ordered_vars_LR)={len(ordered_vars_LR)}')
    print(f'len(ordered_vars_PFI)={len(ordered_vars_PFI)}')

    if FILTER_NUMERIC:
        numeric_features = {'case_weight', 'cmg', 'acute_days', 'alc_days', 'age'}
        numeric_features = set(map(_capitalize_feature_name, numeric_features))
        ordered_vars_LR = list(filter(lambda feature: feature not in numeric_features, ordered_vars_LR))
        # ordered_vars_PFI = list(filter(lambda feature: feature not in numeric_features, ordered_vars_PFI))
        print(f'len(ordered_vars_LR) after removing num = {len(ordered_vars_LR)}')

    LR_rank = {feature:ix+1 for ix, feature in enumerate(ordered_vars_LR)}
    PFI_rank = {feature:ix+1 for ix, feature in enumerate(ordered_vars_PFI)}

    fig, ax = plt.subplots(1, figsize=FIGSIZE)
    ranking1_to_label={}
    ranking2_to_label={}
    for feature in all_features:
        ranking1 = PFI_rank[feature]
        ranking1_to_label[feature_count+1 - ranking1]=feature

        if feature in LR_rank:
            ranking2 = LR_rank[feature]
            ranking2_to_label[feature_count+1 - ranking2]=feature
            x = [1, 2]
            y = [feature_count+1 - ranking1, feature_count+1 - ranking2]
        else:
            x=[1,1]
            y = [feature_count+1 - ranking1,feature_count+1 - ranking1]            
        ax.plot(x, y, marker='o')

        # print(f'feature={str(feature):20} - ranking1={str(ranking1):3} - ranking2={str(ranking2):3}')

    ax.set_yticks(range(1,1+len(all_features)))
    ax.set_yticklabels([ranking1_to_label[ix] for ix in range(1, 1+len(all_features))])
    twin_ax = ax.twinx()

    if FILTER_NUMERIC:
        init_ = len(numeric_features)+1
    else:
        init_ = 1
    twin_ax.set_yticks(range(init_,
                             init_ + len([ix for ix in range(1, 1+len(all_features)) if ix in ranking2_to_label])))
    # twin_ax.set_yticklabels(np.array(range(1,1+len(all_features)))[::-1])
    twin_ax.set_yticklabels([ranking2_to_label[ix] for ix in range(1, 
                                                                   1+len(all_features)) if ix in ranking2_to_label])

    twin_ax.set_ylim(ax.get_ylim())

    ax.set_xticks([1,2], [1, 2])
    ax.set_xticklabels(['PFI Ranking', 'LR Ranking'])


    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    twin_ax.tick_params(axis='both', which='major', labelsize=10)
    twin_ax.tick_params(axis='both', which='minor', labelsize=10)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    print(f'Intersection of feature_names (len)={len(all_features)}')

    fig.savefig(config['feature_ranking_plot'], bbox_inches='tight')
