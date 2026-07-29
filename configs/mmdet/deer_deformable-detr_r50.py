# Deformable DETR R50 on the deer pooled split — the seminal efficient DETR.
# Native schedule is 50 epochs; we keep it (50 < the roster's 70 cap).
_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

data_root = '/work/nvme/bgte/tislam6/wildlife_project/data/dataset/yolo_v3/'
metainfo = dict(classes=('deer',))

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=4, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 ann_file='coco_annotations/train.json',
                 data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=4, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/val.json',
                 data_prefix=dict(img='images/val/')))
test_dataloader = dict(
    batch_size=4, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/test.json',
                 data_prefix=dict(img='images/test/')))
val_evaluator = dict(ann_file=data_root + 'coco_annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_annotations/test.json')

# batch 8 died with CUDA OOM at epoch 5 on a 40 GB A100 (job 20565690): multi-scale
# deformable attention peaks well above its steady-state ~29 GB. Halved to 4, lr scaled
# with it.
# base: lr 2e-4 @ total batch 32 -> linear scale to batch 4
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))
train_cfg = dict(max_epochs=50, val_interval=2)
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=50, by_epoch=True,
         milestones=[40], gamma=0.1),
]
default_hooks = dict(checkpoint=dict(
    interval=10, save_best='coco/bbox_mAP_50', max_keep_ckpts=3))

load_from = '/work/nvme/bgte/tislam6/wildlife_project/weights_mmdet/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth'
