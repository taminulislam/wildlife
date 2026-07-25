# Faster R-CNN R50-FPN on the deer pooled split — classic two-stage baseline.
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

data_root = '/work/nvme/bgte/tislam6/wildlife_project/data/dataset/yolo_v3/'
metainfo = dict(classes=('deer',))

model = dict(
    backbone=dict(init_cfg=None),  # load_from supplies all weights; no net access
    roi_head=dict(bbox_head=dict(num_classes=1)))

train_dataloader = dict(
    batch_size=16, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 ann_file='coco_annotations/train.json',
                 data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=16, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/val.json',
                 data_prefix=dict(img='images/val/')))
test_dataloader = dict(
    batch_size=16, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/test.json',
                 data_prefix=dict(img='images/test/')))
val_evaluator = dict(ann_file=data_root + 'coco_annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_annotations/test.json')

train_cfg = dict(max_epochs=70, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=70, by_epoch=True,
         milestones=[47, 62], gamma=0.1),
]
default_hooks = dict(checkpoint=dict(
    interval=10, save_best='coco/bbox_mAP_50', max_keep_ckpts=3))

load_from = '/work/nvme/bgte/tislam6/wildlife_project/weights_mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
