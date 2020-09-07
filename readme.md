# Wheat Head Detection based on YOLOv3

Currently this repo will summarize and implement current loss functions and non-maximum suppression methods came up for object detection(simalar to the other repo obj-det-loss,but the other repo did not achieve a good result), so I work on this repo first.

Our goal is to analyze different tricks.


+ [x] Revise codes to be more readable and concise
+ [x] Loss_Funcs
  + [x] bbox loss
    + [x] Anchor-based Loss
      + [x] YOLOv3-based
        + [x] Regression Loss
        + [x] IOU Loss
        + [x] GIOU Loss$^{[1]}$#deal with gradient vanish caused by IOU is zero for non-overlap
        + [x] Combined regression with GIOU
  + [x] classification loss
     + [x]dice loss$^{[2]}$ #help deal with class imbalance
+ [x] Non-maximum-suppression
  + [x] Hard NMS
  + [x] Soft NMS$^{[3]}$

[1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
[2]:"v-net loss"
[3]:"Soft-NMS -- Improving Object Detection With One Line of Code":https://arxiv.org/pdf/1704.04503.pdf
