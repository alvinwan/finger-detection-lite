from train import IndexFingerBinDataset
from train import get_index_finger_bin_predictor
from train import bin_to_position
import numpy as np
import torch
import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)

predictor = get_index_finger_bin_predictor()
trainset = IndexFingerBinDataset('data/X_train.npy', 'data/Y_train.npy')
trainloader = iter(torch.utils.data.DataLoader(trainset, batch_size=1))

while True:

    data = next(trainloader)
    im = data['image'].numpy().reshape(90, 160, 3).astype(np.uint8)
    label = np.ravel(data['label'].numpy())

    position = bin_to_position(np.asscalar(label), 160, 90)
    cv2.circle(im, position, 3, GREEN, thickness=-1)

    x, y = predictor(im)
    cv2.circle(im, (x, y), 3, (255, 0, 0), thickness=-1)
    bin_width, bin_height = 40, 30
    for x in range(1, 4):
        cv2.line(im, (x * bin_width, 0), (x * bin_width, 90), RED)
    for y in range(1, 3):
        cv2.line(im, (0, y * bin_height), (160, y * bin_height), RED)

    cv2.imshow('frame', im)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
