"""Helper functions"""
import pandas as pd
import numpy as np
import datetime
from pathlib import Path  


def data_loader():
    df_features = pd.read_csv("./data/dengue_features_train.csv")
    df_test = pd.read_csv("./data/dengue_features_test.csv")
    df_label = pd.read_csv("./data/dengue_labels_train.csv")
    df_train = pd.concat([df_features, df_label.loc[:,'total_cases']], axis=1)
    return df_train, df_test




def get_data_into_submission_format(prediction: np.array) -> None:
    """THis function is used to put the predictions into the right
    submission format

    Args:
        prediction (np.array): THis is an array with the predictions
    """
    subm = pd.read_csv('./submission/submission_format.csv')

    labels = pd.DataFrame({'total_cases': pd.Series(prediction)})
    subm.loc[:, 'total_cases'] = labels.astype(int)

    time = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M")

    
    filepath = Path(f'submission/submission_{time}.csv', index=False)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    subm.to_csv(filepath) 


    # subm.to_csv(f'submission_{time}.csv', index=False)
    # print("Saved!")