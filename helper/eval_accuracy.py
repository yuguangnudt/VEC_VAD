from __future__ import print_function, absolute_import
import numpy as np

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def simple_accuracy(output, target):
    output = output.numpy()
    target = np.squeeze(target.numpy())

    res = np.argmax(output, axis=1)
    correct = np.sum(res == target)
    return correct / target.shape[0]

def binary_accuracy(output, target):
    output = np.squeeze(output.data.cpu().numpy())
    target = np.squeeze(target.cpu().numpy())

    res = np.zeros((output.shape[0],))
    res[output > 0.5] = 1.
    correct = np.sum(res == target)
    return correct / target.shape[0]