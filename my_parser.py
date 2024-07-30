import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

    ##############################
    # Training parameters
    ##############################
    parser.add_argument("--train_batch_size", type=int, default=160,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=80,
                        help="number of epochs to train for") 
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")  
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.") 

    ##############################
    # Inference parameters
    ##############################
    parser.add_argument("--infer_batch_size", type=int, default=300,
                        help="Batch size for inference (caching and testing)")

    ##############################
    # Model parameters
    ############################## 
    parser.add_argument("--backbone_name", type=str, default='dinov2', 
                        choices=["dinov2", "vit_b_16", "vit_b_32", 
                                 "deit_base_patch16_224", "deit3_base_patch16_224", "deit3_base_patch16_224_in21ft1k",
                                 
                                 "resnet50", "resnet101", 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
                                 "mobilenet_v2", "efficientnet",
                                 'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_5','shufflenet_v2_x2_0',
                                ])
    parser.add_argument("--aggregator_name", type=str, default='token_module', 
                        choices=["gemhead", "token_module", "mixvpr"])
    parser.add_argument("--foundation_model_path", type=str, default='pth/dinov2_vitb14_pretrain.pth',
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--num_trainable_blocks", type=int, default=4,
                        help="Number of blocks of backbone training") 
    
    
    ##############################
    # aggregator parameters
    ##############################  
    parser.add_argument("--kernel_size", type=int, default=3) 
    parser.add_argument("--dim", type=int, default=768) 
    parser.add_argument("--nc", type=int, default=4096,
                        help="Output dimension of the aggregator. If None, don't use a fully connected layer.")    
    parser.add_argument("--mlp_ratio", type=float, default=0.75,
                        help="Number of token block") 

    ##############################
    # other parameters
    ##############################    


    args = parser.parse_args() 
    return args

