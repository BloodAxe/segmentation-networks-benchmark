import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    eps = 1e-7
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) + eps
    return 2 * intersection / union


def jaccard_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    eps = 1e-7
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) + eps
    return intersection / (union - intersection)


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)


def smooth_jaccard_loss(y_true, y_pred):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 100

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def bce_jaccard_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
