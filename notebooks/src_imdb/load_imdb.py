import pandas as pd
import numpy as np

def load_imdb(new_data:int = 30_000,base_csv:str = '../data/aclImdb/IMDBDataset.csv'):
    """
    Load the IMDB dataset
    :param base_csv: the path of the dataset file.
    :return: train, validation and test set.
    """
    # Add your code here. 
    imdb_df = pd.read_csv(base_csv)

    train, validate, test = np.split(imdb_df.sample(frac=1, random_state=42), 
                       [int(.7*len(imdb_df)), int(.8*len(imdb_df))])
    
    x_train, y_train = np.array(train.iloc[:new_data, 0]), np.array(train.iloc[:new_data, 1])
    x_valid, y_valid = np.array(validate.iloc[:, 0]), np.array(validate.iloc[:, 1])
    x_test, y_test = np.array(test.iloc[:, 0]), np.array(test.iloc[:, 1])

    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    load_imdb()