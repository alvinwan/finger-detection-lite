"""Prepare the dataset by creating .npy files from images."""

import cv2
import os
import numpy as np


folder = 'data/08667308'
Xs = []
Ys = []
split = 0.8


def main():
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        im = cv2.imread(path)
        Xs.append(np.ravel(im))

        parts = path.split('_')
        x = int(parts[3])
        y = int(parts[5].split('.')[0])
        is_left = int(parts[0] == 'left')

        Ys.append((is_left, x, y))

    X = np.stack(Xs).astype(int)
    Y = np.stack(Ys).astype(int)

    n = int(X.shape[0] * split)
    X_train, X_test = X[:n], X[n:]
    Y_train, Y_test = Y[:n], Y[n:]

    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/Y_train.npy', Y_train)
    np.save('data/Y_test.npy', Y_test)

if __name__ == '__main__':
    main()
