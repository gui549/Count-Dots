# Let's Count Dots

Count how many dots are in an image using Deep Learning.

**Dataset Example**

<kbd><img src = "https://user-images.githubusercontent.com/70506921/131016792-248ebe45-d800-4d2e-8695-4ca5cb7a54b8.png" width="250" height="250"/></kbd><kbd><img src = "https://user-images.githubusercontent.com/70506921/131015688-8e2954b8-d03e-48fd-a7b1-7598bf427c88.png" width="250" height="250"/></kbd>

<kbd><img src = "https://user-images.githubusercontent.com/70506921/131015693-9200e37c-c27b-48b2-ab3c-8963777f347d.png" width="250" height="250"/></kbd><kbd><img src = "https://user-images.githubusercontent.com/70506921/131016797-8236869c-319f-4137-8477-0a824c08c103.png" width="250" height="250"/></kbd>

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

