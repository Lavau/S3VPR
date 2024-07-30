import torch
import logging
logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING
import numpy as np
from tqdm import tqdm 
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import my_parser as parser
import commons 
from net import net   
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities

import warnings
warnings.filterwarnings("ignore")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"



#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()

# 请填写验证集名字
# args.val_set_names = ['msls_val', 'tokyo247', 'SPED', 'nordland', 'pitts30k_val', 'pitts30k_test', 'pitts250k_test', 'pitts250k_val']
# args.val_set_names = ['msls_val', 'tokyo247']
# args.val_set_names = ['pitts30k_test']
args.val_set_names = ['msls_val']

args.save_dir = join("logs", f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{args.backbone_name}-{args.aggregator_name}")
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"validation dataset names:{str(args.val_set_names)}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")



#### Initialize model  
model = net.GeoLocalizationNet(args) 
model = model.to(args.device)
model = torch.nn.DataParallel(model)

# for i in model.children(): 
#     logging.debug(i) 
# for name, param in model.named_parameters():
#     # logging.debug(f"{param.requires_grad} {name}")
#     print(f"{param.requires_grad} {name}")



#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)



#### Getting GSVCities
train_dataset = get_GSVCities()
train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}

val_dataloaders = util.get_val_dataloader(args) 



#### Training loop
msls_val_best_r1 = start_epoch_num = not_improved_num = 0
ds = DataLoader(dataset=train_dataset, **train_loader_config) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds, desc="iteration", ncols=100):       
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1) 
        
        descriptors = model(images.to(args.device))

        descriptors = descriptors.cuda()
        loss = loss_function(descriptors, labels) # Call the loss_function we defined above     
        del descriptors
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls = test.test_val_sets(args, model, val_dataloaders)
    
    msls_val_current_r1 = recalls['msls_val'][1] 
    is_best = msls_val_current_r1 > msls_val_best_r1
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "msls_val_best_r1": msls_val_best_r1,
        "val_set_names": args.val_set_names, "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth"
    )
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous msls-val best R@1 = {msls_val_best_r1:.4f}, current R@1 = {msls_val_current_r1:.4f}")
        msls_val_best_r1 = recalls['msls_val'][1] 
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: msls-val best R@1 = {msls_val_best_r1:.4f}, current R@1 = {msls_val_current_r1:.4f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best  msls-val best R@1 = {msls_val_best_r1:.4f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")






args.val_set_names = ['pitts30k_test']
# args.val_set_names = ['msls_val', 'tokyo247', 'SPED', 'nordland', 'pitts30k_val', 'pitts30k_test', 'pitts250k_test', 'pitts250k_val']
val_dataloaders = util.get_val_dataloader(args)
#### Test best model on test set
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)
test.test_val_sets(args, model, val_dataloaders)



#### Test last model on test set
logging.info("Test *last* model on test set")
last_model_state_dict = torch.load(join(args.save_dir, "last_model.pth"))["model_state_dict"]
model.load_state_dict(last_model_state_dict)
test.test_val_sets(args, model, val_dataloaders)

