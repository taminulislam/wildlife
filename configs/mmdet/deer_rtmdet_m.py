# RTMDet-M on the deer pooled split — mmdetection's real-time SOTA one-stage.
_base_ = 'mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py'

data_root = '/work/nvme/bgte/tislam6/wildlife_project/data/dataset/yolo_v3/'
metainfo = dict(classes=('deer',))

model = dict(
    backbone=dict(init_cfg=None),  # load_from supplies all weights; no net access
    bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=32, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=False, min_size=32),
                 ann_file='coco_annotations/train.json',
                 data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=32, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/val.json',
                 data_prefix=dict(img='images/val/')))
test_dataloader = dict(
    batch_size=32, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='coco_annotations/test.json',
                 data_prefix=dict(img='images/test/')))
val_evaluator = dict(ann_file=data_root + 'coco_annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_annotations/test.json')

# base: 300 epochs @ total batch 256, AdamW lr 0.004 -> ours: 70 epochs @ 32
optim_wrapper = dict(optimizer=dict(lr=0.0005))
max_epochs = 70
train_cfg = dict(max_epochs=max_epochs, val_interval=2, dynamic_intervals=None)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=0.0005 * 0.05, begin=max_epochs // 2,
         end=max_epochs, T_max=max_epochs // 2, by_epoch=True,
         convert_to_iter_based=True),
]
# base switches to the no-mosaic pipeline for the last 20 of 300 epochs; keep the
# same "last 10 epochs" idea at our scale.
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook', switch_epoch=max_epochs - 10,
         switch_pipeline=_base_.train_pipeline_stage2),
]
default_hooks = dict(checkpoint=dict(
    interval=10, save_best='coco/bbox_mAP_50', max_keep_ckpts=3))

load_from = '/work/nvme/bgte/tislam6/wildlife_project/weights_mmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
