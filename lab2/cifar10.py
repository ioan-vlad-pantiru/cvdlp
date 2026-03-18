import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']


def _load_batch(filepath):
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    images = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b'labels'])
    return images, labels


def load_cifar10(root):
    xs, ys = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(os.path.join(root, f'data_batch_{i}'))
        xs.append(imgs)
        ys.append(lbls)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = _load_batch(os.path.join(root, 'test_batch'))
    return X_train, y_train, X_test, y_test


# Alias for the typo present in the notebook cell
load_ciaf10 = load_cifar10


def visualize_images(images, labels, nrows=3, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for ax, img, lbl in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(LABELS[lbl])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
