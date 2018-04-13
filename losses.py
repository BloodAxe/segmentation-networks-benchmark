import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    eps = 1e-15
    union = K.sum(y_true) + K.sum(y_pred) + eps

    return 2.0 * intersection / union


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)

    eps = 1e-15
    union = K.sum(y_true) + K.sum(y_pred) + eps
    return intersection / (union - intersection)


def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)


def bce_jaccard_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def focal_loss(gamma=2., alpha=2.):
    '''
    Compatible with tensorflow backend
    https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
    '''

    def focal_loss_fixed(y_true, y_pred):  # with tensorflow
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())  # improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
