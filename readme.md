# Wheat Head Detection based on YOLOv3

Currently this repo will summarize and implement current loss functions and non-maximum suppression methods came up for object detection(simalar to the other repo obj-det-loss,but the other repo did not achieve a good result), so I work on this repo first.

Our goal is to analyze different tricks.

+ [x] Revise codes to be more readable and concise
+ [x] Loss_Funcs
  + [x] bbox loss
    + [x] Anchor-based Loss
      + [x] YOLOv3-based
        + [x] Regression Loss #testing
        + [x] IOU Loss
        + [x] GIOU Loss$[1]}#deal with gradient vanish caused by IOU is zero for non-overlap
        + [x] Combined regression with GIOU
  + [x] loss for confidence
    + [x] Binary Cross Entropy
      + It is so hard to find a suitble pos/neg weight T T  
    + [x] dice loss[2]
      + hope to help deal with class imbalance
      + not so good as expect
  + [x] Others
    + [x] Use tanh to w,h to avoid grad explosion
    + [x] Sum vs Mean
      + In my opinion, mean is literally better for analyze loss change but also means batch size and number of ground truth will influence gradient
      + even in some extent means errors in the crowded scenes(which is usually harder) get less penalty
      + So I display the mean(for analysis) but optimize on sum loss.
+ [x] Non-maximum-suppression
  + [x] Hard NMS
  + [x] Soft NMS[3]
+ [x] Other Tricks
  + [x] Mosaic Augmentation
  + [x] Sort ground truth to maximize matches reasonablly for multiple matches
  + [ ] Hard Key example mining and other machine learning tricks
+ Results
  + On Validation Set
    + YOLO-SPP
      + YOLO Loss, mAP:0.63

+ Reference:
  + [1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
  + [2]:"v-net loss"
  + [3]:"Soft-NMS -- Improving Object Detection With One Line of Code":https://arxiv.org/pdf/1704.04503.pdf
