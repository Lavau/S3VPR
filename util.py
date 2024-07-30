
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader 

from dataloaders.val import MapillaryValidationDataset
from dataloaders.val import NordlandDataset
from dataloaders.val import PittsburgDataset
from dataloaders.val import Tokyo247Dataset
from dataloaders.val import SPEDDataset


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

 


def get_val_datasets(args):
    valid_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    val_datasets = []
    for name in args.val_set_names:
        if   name.lower() == 'pitts30k_test':
            val_datasets.append(PittsburgDataset.get_30k_test_set(input_transform=valid_transform))
        elif name.lower() == 'pitts30k_val':
            val_datasets.append(PittsburgDataset.get_30k_val_set( input_transform=valid_transform))
        elif name.lower() == 'msls_val':
            val_datasets.append(MapillaryValidationDataset.MSLS(input_transform=valid_transform))
        elif name.lower() == 'pitts250k_test':
            val_datasets.append(PittsburgDataset.get_250k_test_set(input_transform=valid_transform))
        elif name.lower() == 'pitts250k_val':
            val_datasets.append(PittsburgDataset.get_250k_val_set(input_transform=valid_transform))
        elif name.lower() == 'nordland':
            val_datasets.append(NordlandDataset.NordlandDataset(input_transform=valid_transform))
        elif name.lower() == 'tokyo247':
            val_datasets.append(Tokyo247Dataset.Tokyo247Dataset(input_transform=valid_transform))
        elif name.lower() == 'sped':
            val_datasets.append(SPEDDataset.SPEDDataset(input_transform=valid_transform))
        else:
            print(
                f'Validation set {name} does not exist or has not been implemented yet')
            raise NotImplementedError
        logging.info(f"Val set: {val_datasets[-1]}")
    return val_datasets


def get_val_dataloader(args):
    val_datasets = get_val_datasets(args)

    val_dataloaders = []
    valid_loader_config = {
        'batch_size': args.infer_batch_size,
        'num_workers':args.num_workers,
        'drop_last': False,
        'pin_memory': True,
        'shuffle': False
    }
    for val_dataset in val_datasets:
        val_dataloaders.append(DataLoader(dataset=val_dataset, **valid_loader_config))
    return val_dataloaders

