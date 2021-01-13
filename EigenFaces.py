import numpy as np
import itertools
from tqdm import tqdm
import pickle
import sys
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from IPython.display import display, clear_output

from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from additional_function import plot_gallery, plot_eigenfaces


# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=0)
_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
print("Dataset images are at the shape of {}X{}".format(h,w))

# explore data
print(X.shape)
plot_gallery(X[0:12,:], h, w)
plt.hist(X[0])

# center the data
mu = np.average(X, axis=0)
X = X - mu
plt.hist(X[0])

# Use pca
n_components = 1000
pca = PCA(n_components=n_components, whiten=False)
X_pca = pca.fit_transform(X)
# show transformed data
eigenvec_mat = pca.components_
plot_eigenfaces(eigenvec_mat, h, w)

# load my image
img_format = '.jpeg' # change the format according to your image. Do not delete the dot sign before the format name
image = Image.open('Data/me' + img_format)
gray = image.convert('L')
g = gray.resize((w, h))
orig = np.asarray(g).astype('float32')
plt.imshow(orig, cmap=plt.cm.gray)
plt.grid(False)

# process my image
flattened_img = np.asarray(g).astype('float32').flatten()
mu_orig = np.average(flattened_img, axis=0)
flattened_img -= mu_orig

# create U and K
k=900
U = eigenvec_mat[1:k+1]
K = np.inner(U,flattened_img)
