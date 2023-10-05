import pandas as pd
from sklearn.model_selection import train_test_split

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    return df_copy, target

def split_df_items(df,item_start):
   """Extract dataframe based on item start at

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    item_start : str
        Name of item type

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """
   df_item = df[df['item_id'].str.startswith(item_start)]
   return df_item
   

def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : pd.Series
        Target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    """
    
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=test_ratio, random_state=42, stratify=features['item_id'])
    return X_train, y_train, X_val, y_val


def save_sets(df_name, X_train=None, y_train=None, X_val=None, y_val=None, path='../../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np
    if X_train is not None:
      np.save(f'{path}{df_name}_X_train', X_train)
    if X_val is not None:
      np.save(f'{path}{df_name}_X_val',   X_val)
    if y_train is not None:
      np.save(f'{path}{df_name}_y_train', y_train)
    if y_val is not None:
      np.save(f'{path}{df_name}_y_val',   y_val)


def split_save(df,df_name,target_col):
   X,y=pop_target(df, target_col)
   X_train, y_train, X_val, y_val = split_sets_random(X, y, test_ratio=0.2)
   save_sets(df_name,X_train, y_train, X_val, y_val, path='../../data/processed/')

def load_sets_train(df_name,path='../../data/processed/'):
    """Load the different locally save training sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}{df_name}_X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}{df_name}_X_train.npy') else None
    y_train = np.load(f'{path}{df_name}_y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}{df_name}_y_train.npy') else None
    columns = ['item_id', 'store_id', 'is_event','day_of_month','month_of_year','day_of_week']
    X_train = pd.DataFrame(X_train, columns=columns)
    y_train = pd.DataFrame(y_train, columns=['revenue'])
    return X_train, y_train

def load_sets_val(df_name,path='../../data/processed/'):
    """Load the different locally save training sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    """
    import numpy as np
    import os.path

    X_val   = np.load(f'{path}{df_name}_X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}{df_name}_X_val.npy')   else None
    y_val   = np.load(f'{path}{df_name}_y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}{df_name}_y_val.npy')   else None
    columns = ['item_id', 'store_id', 'is_event','day_of_month','month_of_year','day_of_week']
    X_val = pd.DataFrame(X_val, columns=columns)
    y_val = pd.DataFrame(y_val, columns=['revenue'])
    return X_val, y_val


