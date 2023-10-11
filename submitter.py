import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def deep_submitter(model, save_path='data/dl_submission.csv'):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    device = torch.device('cuda')

    y_test = pd.read_csv('data/sample_submission.csv')
    x_test = pd.read_csv('data/test_4dl.csv')
    y_test.loc[x_test.DIST==0, 'CI_HOUR'] = 0
    x_test = x_test.loc[x_test.DIST!=0, :]
    x_test_index = torch.tensor(x_test.index, dtype=torch.int32)

    x_test = torch.FloatTensor(x_test.values)
    test_dataset = TensorDataset(x_test_index, x_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))

    model.eval()
    with torch.no_grad():
        for indices, x in tqdm(test_loader):
            indices = indices.numpy()
            x = x.to(device)
            output = model(x)
            output = output.exp() - 1
            output = output.cpu().numpy()
            y_test.loc[indices, 'CI_HOUR'] = output
    
    y_test.loc[y_test.CI_HOUR<0, 'CI_HOUR'] = 0
    
    if save_path is not None:
        y_test.to_csv(save_path, encoding='UTF-8', index=False)
    
    return y_test


def xgb_submitter(model, save_path='data/xgb_submission.csv'):
    y_test = pd.read_csv('data/sample_submission.csv')
    x_test = pd.read_csv('data/test_preprocessed.csv')
    with open('data/test_zero_indices', 'rb') as f:
        test_zero_indices = pickle.load(f)
    y_test.loc[test_zero_indices, 'CI_HOUR'] = 0

    y_pred = model.predict(x_test.values)
    y_pred = np.power(y_pred, 2)
    y_test.loc[~test_zero_indices, 'CI_HOUR'] = y_pred
    y_test.loc[y_test.CI_HOUR<0, 'CI_HOUR'] = 0

    if save_path is not None:
        y_test.to_csv(save_path, encoding='UTF-8', index=False)
    
    return y_test