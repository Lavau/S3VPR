import torch
import logging
logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING 
from os.path import join 
from datetime import datetime
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import my_parser as parser 
from net import net   
import commons 

import warnings
warnings.filterwarnings("ignore")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

#### Initial setup: parser, logging...
args = parser.parse_arguments() 
start_time = datetime.now()

# 请填写验证集名字
# args.val_set_names = ['msls_val', 'nordland', 'tokyo247', 'pitts30k_val', 'pitts30k_test', 'pitts250k_test', 'pitts250k_val']
# args.val_set_names = ['msls_val', 'pitts30k_val', 'nordland']
args.val_set_names = ['msls_val']
# args.val_set_names = ['msls_val', 'tokyo247']
args.save_dir = join("logs", "eval", f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{args.backbone_name}-{args.aggregator_name}")
commons.setup_logging(args.save_dir) 
 
#### Initialize model  
model = net.GeoLocalizationNet(args) 
model = model.to(args.device)
model = torch.nn.DataParallel(model) 

#### Test best model on test set
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join("pth/bs=160_p=3_r=0.75_d=4096_best_model.pth"))["model_state_dict"]
sd = {}
for k in best_model_state_dict.keys(): 
    sd[k] = best_model_state_dict[k] 
model.load_state_dict(sd)

model.eval()

val_dataloaders = util.get_val_dataloader(args) 
test.test_val_sets(args, model, val_dataloaders)

#### Test last model on test set
# logging.info("Test *last* model on test set")
# last_model_state_dict = torch.load(join(args.save_dir, "last_model.pth"))["model_state_dict"]
# model.load_state_dict(last_model_state_dict)
# test.test_val_sets(args, model, val_dataloaders)

# CUDA_VISIBLE_DEVICES=0,1 python eval.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 

