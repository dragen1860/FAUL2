
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition



class VisualH:

    def __init__(self):

        plt.ion()

        self.tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    def update(self, h_spt0, h_spt1, h_qry0, h_qry1, y_spt, y_qry):
        """

        :param h_spt0: [b, h_c, h_d, h_c], hidden representation of spt before update,
        :param h_spt1: hidden representation of spt after update,
        :param h_qry0:
        :param h_qry1:
        :param y_spt: [b]
        :param y_qry: [b]
        :return:
        """
        h_spt0 = h_spt0.cpu().numpy().view(h_spt0.size(0), -1)
        h_spt1 = h_spt1.cpu().numpy().view(h_spt1.size(0), -1)
        h_qry0 = h_qry0.cpu().numpy().view(h_qry0.size(0), -1)
        h_qry1 = h_qry1.cpu().numpy().view(h_qry1.size(0), -1)
        y_spt = y_spt.cpu().numpy()
        y_qry = y_qry.cpu().numpy()

        # [b, -1] => [b, 2]
        h_spt0 = self.tsne.fit_transform(h_spt0)
        h_spt1 = self.tsne.fit_transform(h_spt1)
        h_qry0 = self.tsne.fit_transform(h_qry0)
        h_qry1 = self.tsne.fit_transform(h_qry1)

        self.plot(h_spt0, y_spt, 0, 'h_spt0')
        self.plot(h_spt1, y_spt, 1, 'h_spt1')
        self.plot(h_qry0, y_qry, 2, 'h_qry0')
        self.plot(h_qry1, y_qry, 3, 'h_qry1')

        plt.pause(0.001)

    def plot(self, X, y, fig, title):
        """

        :param X:
        :param y:
        :param fig: figure id
        :param title:
        :return:
        """
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(fig)
        ax = plt.subplot(111)
        # for each point
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'normal', 'size': 10})

        plt.xticks([]), plt.yticks([])
        plt.title(title)





# Scale and visualize the embedding vectors
def plot_embedding(X, y, digits, title=None, annotation=True):
    """

    :param X: [b, features]
    :param title:
    :return:
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    # for each point
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'normal', 'size': 10})

    if hasattr(offsetbox, 'AnnotationBbox') and annotation:
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def main():
    # 8x8
    digits = datasets.load_digits(n_class=7)
    # [720, 64]
    X = digits.data
    # [720]
    y = digits.target

    # [720, 64]
    # print(type(X), X.shape)
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca, y, digits,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0), annotation=False)


    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    print(X_tsne.shape)

    plot_embedding(X_tsne, y, digits,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0), annotation=False)

    plt.show()

if __name__ == '__main__':
    main()