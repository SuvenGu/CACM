import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from einops import rearrange

mean=np.array([0.0592465,0.08257256,0.07940245,0.122184,0.26104507,0.32686248,0.33226496,0.35085383,0.25254872,0.16759971])
std = np.array([0.03050405, 0.03372968, 0.05003787, 0.05067676, 0.07917259,0.11817433, 0.11635097, 0.12101864, 0.09216624, 0.08916556])

# ## tmn tmx srad
mean_c=np.array([51.9398428,  169.45926927, 1877.14472879])
std_c = np.array([115.44607918 , 123.6412339, 718.64770224])

    
class CropAttriMappingDataset(Dataset):
    """
    crop classification dataset
    """
    def __init__(self, path,c_dim=14,T=10,T_a=6):
        dfs= []
        if isinstance(path, list):
            for i in path:
                print(i)
                df = pd.read_csv(i)
                dfs.append(df)
            arrays = [df.values for df in dfs]
            data = np.concatenate(arrays, axis=0)
        else:
            data = pd.read_csv(i)
            data = np.array(data.values)

        print("data loaded!")
        print(data.shape)
        x = data[:,1:]
        self.x = x[:,:100]
        self.cond = x[:,100:]
        self.y = data[:,0].astype("int64")

        self.x =rearrange(self.x,"b (t c)->b t c",t = T)
        self.cond =rearrange(self.cond,"b (t c)->b t c",c =14 )
 
        self.x = (self.x - mean)/std

        ind = [9,10,7]
        self.cond = self.cond[:,:,ind]
        self.cond = (self.cond - mean_c)/std_c

        self.x =rearrange(self.x,"b t c->b c t").astype("float32")
        self.cond =rearrange(self.cond,"b t c->b c t").astype("float32")
       
        # for tsne vis
        torch.manual_seed(50)
        new_idx = torch.randperm(self.x.shape[0])
        self.x = self.x[new_idx]
        self.y = self.y[new_idx]
        self.cond = self.cond[new_idx]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx],self.cond[idx]
