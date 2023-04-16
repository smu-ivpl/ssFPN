## YOLOR + S^2 Feature


### Environment
- Ubuntu 18.04 with Tesla V100
- torch 1.9.0
- torchvision 0.10.0
```
pip install -r requirements.txt
```
[Download other pre-trained models.](https://drive.google.com/drive/folders/17pTfcEsxAX6YGbuclQJqGCJ-_NvAznk2?usp=share_link)

### Test COCO validation
```
python test.py --data data/coco.yaml --img 1280 --batch 32 --conf 0.001 --iou 0.65 --device 0 --name [name]
```

### Training

```
# yolov4-p6-s2
python -m torch.distributed.launch --nproc_per_node 3 --master_port 9527 train.py --batch-size 18 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6-sequence.yaml --sync-bn --device 0,1,2 --name [name] --hyp hyp.scratch.1280.yaml --epochs 300 --resume
python -m torch.distributed.launch --nproc_per_node 3 --master_port 9527 tune.py --batch-size 18 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6-sequence.yaml --weights runs/train/[name]/weights/last_298.pt --sync-bn --device 0,1,2 --name [project name] --hyp hyp.finetune.1280.yaml --epochs 450
```

