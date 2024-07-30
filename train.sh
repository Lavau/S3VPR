CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75
# CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=3
# CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=3 --train_batch_size=170
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=gemhead
# space + gemhead
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2  --aggregator_name=token_module_gai --mlp_ratio=0.75 
# channel + gemhead
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module_gai --mlp_ratio=0.75 


################################################
# 骨干网验证
################################################
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=vit_b_16 --aggregator_name=gemhead
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=vit_b_16 --aggregator_name=token_module --mlp_ratio=0.75 
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=vit_b_32 --aggregator_name=gemhead
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=vit_b_32 --aggregator_name=token_module --mlp_ratio=0.75 
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=deit3_base_patch16_224 --aggregator_name=gemhead 
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=deit3_base_patch16_224 --aggregator_name=token_module --mlp_ratio=0.75 
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=deit_base_patch16_224  --aggregator_name=gemhead 
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=deit_base_patch16_224  --aggregator_name=token_module --mlp_ratio=0.75  
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=deit3_base_patch16_224_in21ft1k  --aggregator_name=gemhead 
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=deit3_base_patch16_224_in21ft1k  --aggregator_name=token_module --mlp_ratio=0.75
 
 



# CNN 模型
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=efficientnet --aggregator_name=token_module --mlp_ratio=0.75 --dim=1280
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=mobilenet_v2 --aggregator_name=token_module --mlp_ratio=0.75 --dim=1280

CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=shufflenet_v2_x1_0 --aggregator_name=token_module --dim=1024
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=shufflenet_v2_x0_5 --aggregator_name=token_module --dim=1024
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=shufflenet_v2_x2_0 --aggregator_name=token_module --dim=2048
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=shufflenet_v2_x1_5 --aggregator_name=token_module --dim=1024

CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=resnext50_32x4d --aggregator_name=token_module --dim=2048
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=resnext101_32x8d --aggregator_name=token_module --dim=2048 --train_batch_size=80
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=resnext101_64x4d --aggregator_name=token_module --dim=2048 --train_batch_size=80

CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=resnet50 --aggregator_name=gemhead --dim=2048
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=resnet50 --aggregator_name=token_module --mlp_ratio=0.75 --dim=2048
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=resnet101 --aggregator_name=gemhead --dim=2048 --train_batch_size=150
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=resnet101 --aggregator_name=token_module --mlp_ratio=0.75 --dim=2048 --train_batch_size=150



################################################
# kernel size
################################################
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 --kernel_size=1
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2 --aggregator_name=token_module --mlp_ratio=0.75 --kernel_size=5 --train_batch_size=90
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2 --aggregator_name=token_module --mlp_ratio=0.75 --kernel_size=7 --train_batch_size=50

################################################
# mlp ratio
################################################
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.25
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.5
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=1
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=2
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=3
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=4


################################################
# nc
################################################
CUDA_VISIBLE_DEVICES= python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 --nc=512
CUDA_VISIBLE_DEVICES= python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 --nc=1024
CUDA_VISIBLE_DEVICES= python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 --nc=2048
CUDA_VISIBLE_DEVICES= python train.py --backbone_name=dinov2  --aggregator_name=token_module --mlp_ratio=0.75 --nc=8192



################################################
# supplementary material
################################################
CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone_name=dinov2  --aggregator_name=mixvpr
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone_name=dinov2  --aggregator_name=mixvpr_token_block
CUDA_VISIBLE_DEVICES=4,5 python train.py --backbone_name=dinov2  --aggregator_name=ssm --train_batch_size=150