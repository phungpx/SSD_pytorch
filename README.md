# Single Shot MultiBox Detector

## Dataset
|ID|Dataset Name|Train|Val|Test|Format|
|:--:|:--------:|:--------:|:--:|:--:|:--:|
1|Pascal VOC 2007 |5,011|4,952|-|PASCAL XML|
2|Pascal VOC 2012 |1,464|1,449|-|PASCAL XML|

## Main Functions
Dataset: ```flame/core/data/pascal_dataset.py``` \
Model: ```flame/core/model``` \
Evaluator: ```flame/handlers/metrics/mean_average_precision``` \
Visualization: ```flame/handlers/region_predictor.py```

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

### Examples
| SSD300 |
| --- |
|<img src="https://user-images.githubusercontent.com/61035926/136917126-1858e4ed-b320-4bc6-9bda-e7108c9e5899.jpg" width="300">|
|<img src="https://user-images.githubusercontent.com/61035926/136917309-788c402e-06be-4ed1-a553-b1e79d59de19.jpg" width="300">|
|<img src="https://user-images.githubusercontent.com/61035926/136917607-abc1ac67-e4b3-4ae9-a2d4-53cbd77ae2a9.jpg" width="300">|
|<img src="https://user-images.githubusercontent.com/61035926/136917805-b9ff4430-d97b-4aa5-a26e-c9cf2fb671e5.jpg" width="300">|
