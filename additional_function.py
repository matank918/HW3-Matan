import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

def plot_gallery(images, h, w,rows=3, cols=4):
    plt.figure(figsize=(10,10))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i, :].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def plot_eigenfaces(eigenvec_mat, h, w, rows=3, cols=4):
    plt.figure(figsize=(10, 10))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(eigenvec_mat[i, :].reshape((h, w)), cmap=plt.cm.gray)
        plt.title("$u_{" + str(i + 1) + "}$")
        plt.xticks(())
        plt.yticks(())
    plt.show()


def plot_result(K, U, mu_orig, c, h, w):
    s = np.zeros((2914,))
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    for j in range(K):
        s += c[j] * U[j, :]
        if np.mod(j, 10) == 0:
            axes[0].imshow(U[j, :].reshape((h, w)), cmap=plt.cm.gray)
            axes[0].grid(False)
            corrected_image = s + mu_orig
            axes[1].imshow(corrected_image.reshape((h, w)), cmap=plt.cm.gray)
            axes[1].grid(False)
            if c[j] < 0:
                axes[0].set_title('{:.2f}'.format(c[j]))
            else:
                axes[0].set_title('+{:.2f}'.format(c[j]))
            display(fig)
            clear_output(wait=True)
            plt.pause(0.3)
            if c[j] > 0:
                p = '+' + str(c[j])
            else:
                p = str(c[j])