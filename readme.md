# A Coarse-to-fine Wheat Head Detector with Intermediate Supervision

- v1: with heatmap as intermediate supervision: seems not work
- v2: Resnet with YOLOv3 anchor-based head
- v3: rewrite network as U-shape cascade YOLO
- To do
  - [x] Metrics:precision,recall and AP
  - [x] trainer.py
    - [x] train
    - [x] validate
    - [x] load_epoch
    - [x] save_epoch
    - [ ] make_prediction
  - [ ] NMS
    - [x] soft-NMS
  - [ ] New model with Attention network
  - [ ] With U-net as Backbone
  - [x] tensorboard 
  - [ ] visualization
  - [ ] Paper
