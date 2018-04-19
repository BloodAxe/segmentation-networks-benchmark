import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib import numpy_losses

sns.set()


def main():
    steps = 10
    loss_functions = [numpy_losses.binary_crossentropy,
                      numpy_losses.jaccard_loss,
                      numpy_losses.smooth_jaccard_loss,
                      numpy_losses.bce_smooth_jaccard_loss]

    for loss_fn in loss_functions:
        y_true = np.ones((224, 224), dtype=np.float32)
        y_pred = y_true.copy()

        losses = [loss_fn(y_true, y_pred)]

        for pred_val in range(0, 1000):
            y_pred[...] = 1 - pred_val / 1000
            loss_val = loss_fn(y_true, y_pred)
            losses.append(loss_val)

        #
        # for row in range(0, 224):
        #     for col in range(0, 224):
        #         y_pred[row, col] = 0.5
        #         loss_val = loss_fn(y_true, y_pred)
        #         losses.append(loss_val)

        plt.figure()
        plt.title(loss_fn.__name__)
        plt.plot(losses)
        plt.ylabel('Loss value')
        plt.xlabel('Wrong pixels')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
