from train import IndexFingerBinDataset
from train import get_index_finger_bin_predictor
from train import bin_to_position
import numpy as np
import torch
import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)

predictor = get_index_finger_bin_predictor()
testset = IndexFingerBinDataset('data/X_test.npy', 'data/Y_test.npy')
testloader = iter(torch.utils.data.DataLoader(testset, batch_size=1))

image_width = 160
image_height = 90
xbins = 4
ybins = 3
bin_width = image_width // xbins
bin_height = image_height // ybins

while True:

    data = next(testloader)
    im = data['image'].numpy().reshape(image_height, image_width, 3).astype(np.uint8)
    label = np.ravel(data['label'].numpy())

    position = bin_to_position(np.asscalar(label), image_width, image_height)
    cv2.circle(im, position, 3, GREEN, thickness=-1)

    x, y = predictor(im)
    cv2.circle(im, (x, y), 3, (255, 0, 0), thickness=-1)
    for x in range(1, xbins):
        cv2.line(im, (x * bin_width, 0), (x * bin_width, image_height), RED)
    for y in range(1, ybins):
        cv2.line(im, (0, y * bin_height), (image_width, y * bin_height), RED)

    cv2.imshow('frame', im)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
