# Single Shot MultiBox Detector

## Dataset
|ID|Dataset Name|Train|Val|Test|Format|
|:--:|:--------:|:--------:|:--:|:--:|:--:|
1|Pascal VOC 2007 |5,011|4,952|-|PASCAL XML|
2|Pascal VOC 2012 |1,464|1,449|-|PASCAL XML|

## Usage
### Training
```bash
CUDA_VISIBLE_DEVICES=<x> python -m flame configs/ssd300_vgg16_voc_training.yaml
```
### Testing
```bash
CUDA_VISIBLE_DEVICES=<x> python -m flame configs/ssd300_vgg16_voc_testing.yaml
```

## Performance
### SSD300
![SSD](https://user-images.githubusercontent.com/61035926/136876543-c83bcb5f-99da-438b-b4d7-d5a14ceb7039.png)
