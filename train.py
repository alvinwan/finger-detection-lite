"""Simple, generic object detection pipeline.

Only accepts and detects one object, a cartesian coordinate. The network
additionally outputs confidence for each detection.
"""

from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import argparse
import cv2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.pool = nn.MaxPool2d(2, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 2, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class IndexFingerDataset(Dataset):

    def __init__(self, sample_path: str, label_path: str):
        """
        Args:
            sample_path: Path to `.npy` file containing samples nxd.
            label_path: Path to `.npy` file containign labels nx1.
        """
        self._samples = np.load(sample_path)
        self._labels = np.load(label_path)[:, 1:]
        self._samples = self._samples.reshape((-1, 3, 160, 90))
        self._labels[:, 0] = self._labels[:, 0] / 160
        self._labels[:, 1] = self._labels[:, 1] / 90

        X_shape, Y_shape = self._samples.shape, self._labels.shape
        assert X_shape[0] == Y_shape[0], (X_shape, Y_shape)

        self.X = Variable(torch.from_numpy(self._samples)).float()
        self.Y = Variable(torch.from_numpy(self._labels)).float()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {'image': self._samples[idx], 'label': self._labels[idx]}


def evaluate(outputs: Variable, labels: Variable, normalized: bool=True,
             tol: float=0.05) -> float:
    """Evaluate neural network outputs against labels."""
    Y = labels.data.numpy()
    Yhat = outputs.data.numpy()
    denom = Y.shape[0] if normalized else 1
    err = Yhat - Y
    return float(np.sum(err < tol) / (2 * denom))


def batch_evaluate(net: Net, dataset: Dataset, batch_size: int=500) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = 0.0
    n = dataset.X.shape[0]
    for i in range(0, n, batch_size):
        x = dataset.X[i: i + batch_size]
        y = dataset.Y[i: i + batch_size]
        score += evaluate(net(x), y, False)
    return score / n


def save_state(epoch: int, net: Net, optimizer):
    """Save the state of training."""
    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'outputs/checkpoint.pth')


def get_index_finger_predictor(model_path='assets/model_best.pth'):
    """Returns predictor, from image to emotion index."""
    net = Net().float()
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['state_dict'])

    def predictor(image: np.array):
        """Translates images into emotion indices."""
        h, w, _ = image.shape
        frame = cv2.resize(image, (160, 90)).reshape((1, 3, 160, 90))
        X = Variable(torch.from_numpy(frame)).float()
        ph, pw = np.ravel(net(X).data.numpy())
        return int(ph * h), int(pw * w)
    return predictor


def train(
        net: Net,
        trainset: IndexFingerDataset,
        testset: IndexFingerDataset,
        pretrained_model: dict={}):
    """Main training loop and optimization setup."""
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    best_test_acc = 0

    def status_update(outputs: Variable, labels: Variable):
        """Print train, validation accuracies along with current loss."""
        nonlocal best_test_acc

        train_acc = evaluate(outputs, labels)
        test_acc = evaluate(net(testset.X), testset.Y)
        print('[%d, %5d] loss: %.2f train acc: %.2f val acc: %.2f' %
              (epoch + 1, i + 1, running_loss / i, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_state(epoch, net, optimizer)

    start_epoch = pretrained_model.get('epoch', 0)
    for epoch in range(start_epoch, start_epoch + 20):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = Variable(data['image'].float())
            labels = Variable(data['label'].float())
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9:
                status_update(outputs, labels)


def main():
    """Main script for Index Finger Detection dataset neural network"""
    args = argparse.ArgumentParser('Main training script for finger detection')
    args.add_argument('action', choices=('train', 'eval'),
                      help='Script utility to invoke')
    args.add_argument('--model', help='Path to model to restore from.')
    args.add_argument('--max-batch-size', default=500, type=int,
                      help='Maximum number of samples to pass through network '
                           'due to memory constraints')
    args = args.parse_args()

    trainset = IndexFingerDataset('data/X_train.npy', 'data/Y_train.npy')
    testset = IndexFingerDataset('data/X_test.npy', 'data/Y_test.npy')
    net = Net().float()

    pretrained_model = {}
    if args.model:
        pretrained_model = torch.load(args.model)
        net.load_state_dict(pretrained_model['state_dict'])

    if args.action == 'train':
        train(net, trainset, testset, pretrained_model)
        print('=' * 10, 'Finished Training', '=' * 10)
    elif not args.model:
        raise UserWarning('Need a model to evaluate! Otherwise, you would be '
                          'evaluating a random initialization. Use the --model'
                          'flag.')

    train_acc = batch_evaluate(net, trainset, batch_size=args.max_batch_size)
    print('Training accuracy: %.3f' % train_acc)
    test_acc = batch_evaluate(net, testset, batch_size=args.max_batch_size)
    print('Validation accuracy: %.3f' % test_acc)


if __name__ == '__main__':
    main()