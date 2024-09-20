import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_data_ml100k(data_dir:str)->tuple[pd.DataFrame,int,int]:
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

def split_data_ml100k(data: pd.DataFrame, num_users: int,
                      split_mode: str = 'random', test_ratio: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets in either random mode or sequence-aware mode.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing user-item interactions. Expected columns are user_id, item_id, rating, and timestamp.
    num_users : int
        The number of unique users in the dataset.
    split_mode : str, optional
        The mode of splitting the data. Can be 'random' or 'seq-aware'. Default is 'random'.
    test_ratio : float, optional
        The ratio of the dataset to be used as the test set. Only used if split_mode is 'random'. Default is 0.1.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and testing datasets as pandas DataFrames.
    """
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data: pd.DataFrame, num_users: int, num_items: int, feedback: str = 'explicit') -> tuple[list[int], list[int], list[int], np.ndarray | dict]:
    """
    Load the dataset and convert it into user-item interaction matrices.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing user-item interactions. Expected columns are user_id, item_id, and rating.
    num_users : int
        The number of unique users in the dataset.
    num_items : int
        The number of unique items in the dataset.
    feedback : str, optional
        The type of feedback. Can be 'explicit' or 'implicit'. Default is 'explicit'.

    Returns:
    --------
    tuple[list[int], list[int], list[int], np.ndarray | dict]
        A tuple containing:
        - users: List of user indices.
        - items: List of item indices.
        - scores: List of scores (ratings or implicit feedback).
        - inter: User-item interaction matrix. If feedback is 'explicit', this is a 2D numpy array.
                 If feedback is 'implicit', this is a dictionary where keys are user indices and values are lists of item indices.
    """
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

def split_and_load_ml100k_pytorch(data_dir:str,split_mode: str = 'seq-aware', feedback: str = 'explicit',
                                  test_ratio: float = 0.1, batch_size: int = 256)->tuple[int,int,DataLoader,DataLoader]:
    """
    Split the dataset and load it into PyTorch DataLoader objects.

    Parameters:
    -----------
    split_mode : str, optional
        The mode of splitting the data. Can be 'random' or 'seq-aware'. Default is 'seq-aware'.
    feedback : str, optional
        The type of feedback. Can be 'explicit' or 'implicit'. Default is 'explicit'.
    test_ratio : float, optional
        The ratio of the dataset to be used as the test set. Only used if split_mode is 'random'. Default is 0.1.
    batch_size : int, optional
        The size of the batches for the DataLoader. Default is 256.

    Returns:
    --------
    tuple[int, int, DataLoader, DataLoader]
        A tuple containing:
        - num_users: The number of unique users.
        - num_items: The number of unique items.
        - train_iter: DataLoader for the training set.
        - test_iter: DataLoader for the test set.
    """
    data, num_users, num_items = read_data_ml100k(data_dir)
    train_data, test_data = split_data_ml100k(
        data, num_users, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    
    # Convert to PyTorch tensors
    train_u_tensor = torch.tensor(train_u, dtype=torch.long)
    train_i_tensor = torch.tensor(train_i, dtype=torch.long)
    train_r_tensor = torch.tensor(train_r, dtype=torch.float)
    test_u_tensor = torch.tensor(test_u, dtype=torch.long)
    test_i_tensor = torch.tensor(test_i, dtype=torch.long)
    test_r_tensor = torch.tensor(test_r, dtype=torch.float)
    
    # Create TensorDataset
    train_set = TensorDataset(train_u_tensor, train_i_tensor, train_r_tensor)
    test_set = TensorDataset(test_u_tensor, test_i_tensor, test_r_tensor)
    
    # Create DataLoader
    train_iter = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_iter = DataLoader(test_set, batch_size=batch_size)
    
    return num_users, num_items, train_iter, test_iter
