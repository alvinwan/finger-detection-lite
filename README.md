# Background Agnostic Finger Detection

Index finger detection for the ☝️ gesture across different backgrounds and environments. The particular pretrained model and data in this repository is fitted to one background, but the idea follows: this detector does not require an abnormal setup where *only* a hand is on camera. I use my bedroom as an example.

This repository was built to be lightweight: the neural network here can be trained in a matter of minutes on a standard consumer laptop, and the data likewise can be collected in 5-10 minutes. With extra bells and whistles--more data or a smarter neural network design--this detector could be used more generically. I reduced the problem to a 12-way classification problem, where the neural network simply predicts which sector of the frame, the hand currently points to.

## Installation

1. [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

2. Start by installing [PyTorch](http://pytorch.org).

3. Install all Python dependencies.

```
pip install -r requirements.txt
```

> For Ubuntu users, install `libsm6` for OpenCV to work:

```
apt install libsm6
```

Installation is complete. To get started, launch the demo. Curl your hand into a fist, then use your index finger to point straight up. Move your hand around, and if all works, a red dot will indicate which sector your hand currently points to.

```
python demo.py
```

## Dataset

I made a simple utility for collecting data. The script moves a red dot around the screen, and your job is to follow that dot with your index finger. Every half a second, the script captures the image. Given it pre-determined the red dot's location, your data is automatically labeled. All it takes is a 5-minute or so capture, for approximately 600 samples.

To collect data, start by launching the script. Within several seconds, the red dot will begin to move. At which point, you should trace the red dot with your index finger. Once you're happy with the amount of data collected, hit `q` to close the frame.

```
python utils/collect_uniform.py
```

> You may alternatively use `utils/collect_bounce.py`. The former script ensures the samples are less biased towards the center of the frame.

After collection, use the `prepare.py` script to put all data into numpy matrices.

```
python utils/prepare.py
```

If you'd like to reproduce my results or simply use the data I collected, [download data I used](https://drive.google.com/file/d/1nf0QcDijw6NTVR7Q9qwkWZEcMcCkNnJy/view?usp=sharing) from Google Drive.

## Training

To launch training, use the `main.py` script. The default model and dataset regresses to the exact cartesian coordinates.

```
python main.py train
```

To use the classification approach, and as a result, tradeoff between accuracy and compute power available, use the `--dataset` flag. The variables `xbins` and `ybins` in the source allow you to tune the granularity of our bins.

```
python main.py train --dataset=classification
```

The `--lr` flag allows you to change the learning rate, and the `--model` flag allows you to pass in pretrained models. By default, checkpoints are saved at `outputs/checkpoint.pth`. For my model, I trained at a learning rate of `0.001` for the first 20 epochs and `0.0001` for an additional 20 epochs. 