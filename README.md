# Let's Count Dots

Count how many dots are in an image using Deep Learning.

**Dataset Example**
<img src = "https://user-images.githubusercontent.com/70506921/131015684-5f59518e-4ab5-4d8b-bdc4-8a8a6a4ca7c2.png" width="300" height="300"/><img src = "https://user-images.githubusercontent.com/70506921/131015688-8e2954b8-d03e-48fd-a7b1-7598bf427c88.png" width="300" height="300"/><img src = "https://user-images.githubusercontent.com/70506921/131015693-9200e37c-c27b-48b2-ab3c-8963777f347d.png" width="300" height="300"/>

## Installation
```
pip install -r requirements.txt
```

## Train
**Example**
```
./train.py --root ./datasets/DotsEven/ -f base -l -m resnet_scalar --batch_size 32
```

## Test
**Example**
```
./test.py -m resnet_scalar --load_path ./experiments/base_110.pth --test_path ./datasets/TestDots/
```

