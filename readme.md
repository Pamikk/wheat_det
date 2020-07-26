# A Coarse-to-fine Wheat Head Detector with Intermediate Supervision

- v1: with heatmap as intermediate supervision: seems not work
- v2: introduce attention map and rewrite network as U-shape
- v3: cascade YOLO
- To do
  - [x] Metrics:precision,recall and AP
  - [ ] trainer.py
    - [x] train
    - [x] validate
    - [x] load_epoch
    - [x] save_epoch
    - [ ] make_prediction
  - [ ] NMS
    - [x] soft-NMS
  - [x] tensorboard 
  - [ ] visualization
  - [ ] Paper
