import numpy as np


def binary_crossentropy(y_true, y_pred):
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = - np.sum(np.log(y_pred) * y_true)
    return np.mean(loss)


def jaccard_coef(y_true, y_pred):
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    eps = 1e-7
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) + eps
    return intersection / (union - intersection)


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
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    smooth = 100

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    jac = (intersection + smooth) / (union - intersection + smooth)
    return (1 - jac) * smooth

def bce_jaccard_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)


def bce_smooth_jaccard_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + smooth_jaccard_loss(y_true, y_pred)
