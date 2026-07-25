# DINO-4scale R50 on the deer pooled split — transformer detection SOTA line.
# DINO's design converges in 12-36 epochs (its whole point); 36e is its published
# long schedule, so we use that instead of 70 (70 would also blow the walltime).
_base_ = 'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'

data_root = '/work/nvme/bgte/tislam6/wildlife_project/data/dataset/yolo_v3/'
metainfo = dict(classes=('deer',))

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=4, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 ann_file='coco_annotations/train.json',
                 data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=4, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/val.json',
                 data_prefix=dict(img='images/val/')))
test_dataloader = dict(
    batch_size=4, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/test.json',
                 data_prefix=dict(img='images/test/')))
val_evaluator = dict(ann_file=data_root + 'coco_annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_annotations/test.json')

# base: lr 1e-4 @ total batch 16 -> linear scale to batch 4
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))
train_cfg = dict(max_epochs=36, val_interval=2)
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True,
         milestones=[30], gamma=0.1),
]
default_hooks = dict(checkpoint=dict(
    interval=6, save_best='coco/bbox_mAP_50', max_keep_ckpts=3))

load_from = '/work/nvme/bgte/tislam6/wildlife_project/weights_mmdet/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
