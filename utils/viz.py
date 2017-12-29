"""Visualize the red dot for a given image."""

import cv2


path = 'data/0767/right_000012_x_143_y_64.png'
im = cv2.imread(path)
parts = path.split('_')
x = int(parts[3])
y = int(parts[5].split('.')[0])
cv2.circle(im, (x, y), 3, (0, 0, 255))

while True:
    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
