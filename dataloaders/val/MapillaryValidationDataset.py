from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
# DATASET_ROOT = '../data/mapillary/'
DATASET_ROOT = '/data/wuchenyu/pr/msls/' # 改动

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

class MSLS(Dataset):
    def __init__(self, input_transform = None):
        
        self.input_transform = input_transform
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_dbImages.npy')
        
        # hard coded query image names.
        self.qImages = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_qImages.npy')
        
        # hard coded index of query images
        self.qIdx = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_qIdx.npy')
        
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_pIdx.npy', allow_pickle=True)
        
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def __repr__(self):
        return  (f"< {self.__class__.__name__}, {'msls-val'} - #database: {self.num_references}; #queries: {self.num_queries} >")
    


if __name__ == '__main__':
    # db = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_dbImages.npy')
    # print(db[0], '\n')
        
    # hard coded query image names.
    q = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_qImages.npy')
    print(len(q), q[0], '\n')
    
    # hard coded index of query images
    qid = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_qIdx.npy')
    print(len(qid), qid[:10], '\n')
    
    # hard coded groundtruth (correspondence between each query and its matches)
    pid = np.load('/home/wuchenyu/S2VPR/dataloaders/datasets/msls_val/msls_val_pIdx.npy', allow_pickle=True)
    print(len(qid), pid[0], '\n')