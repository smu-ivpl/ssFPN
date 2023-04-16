## 2-stage models
- Faster-RCNN+S^2
- Mask-RCNN+S^2
- Casacade-RCNN+S^2

-----
### Environment
- Ubuntu 18.04 with Tesla V100
- torch 1.8.1
- torchvision 0.9.1
```
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# https://github.com/facebookresearch/detectron2

pip install opencv-python
pip install pyyaml==5.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -e detectron2
```

### COCO Dataset
- https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md   
```
export DETECTRON2_DATASETS=[coco path]
```


### Test COCO validation
- [Download other pre-trained models.](https://drive.google.com/drive/folders/17pTfcEsxAX6YGbuclQJqGCJ-_NvAznk2?usp=share_link)
```
DETECTRON2_DATASETS=[coco path] python plain_train_net.py --config-file configs/S2/faster_rcnn_R_50_FPN_S2_3x.yaml --eval-only MODEL.WEIGHTS ckp/faster-rcnn-r50-fpn-3x-40.2.pkl
```


### Training

```
DETECTRON2_DATASETS=[coco path] python plain_train_net.py --config-file configs/S2/faster_rcnn_R_50_FPN_S2_3x.yaml --num-gpus 4 SEED 42
```

