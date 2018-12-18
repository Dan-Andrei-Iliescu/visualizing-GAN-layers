import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

color1 = '#ee4433'
color2 = '#ee6622'
color3 = '#ff8811'
color4 = '#ffbb00'
color5 = '#aabb00'
color6 = '#77bb44'
color7 = '#44bb88'
color8 = '#4488ff'
color9 = '#6633bb'
color10 = '#ee4488'


def visualize_mnist(X, y, title):
    path = os.path.join('visualizations', title)

    X = X.reshape(X.shape[0], -1)

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(X.reshape(X.shape[0], -1))
    print(
        'Cumulative explained variation for 50 principal components: {}'
        .format(np.sum(pca_50.explained_variance_ratio_))
    )

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done!')

    one = tsne_pca_results[:, 0]
    two = tsne_pca_results[:, 1]

    plt.style.use('ggplot')
    plt.scatter(one[y == 0], two[y == 0], alpha=0.9, color=color1)
    plt.scatter(one[y == 1], two[y == 1], alpha=0.9, color=color2)
    plt.scatter(one[y == 2], two[y == 2], alpha=0.9, color=color3)
    plt.scatter(one[y == 3], two[y == 3], alpha=0.9, color=color4)
    plt.scatter(one[y == 4], two[y == 4], alpha=0.9, color=color5)
    plt.scatter(one[y == 5], two[y == 5], alpha=0.9, color=color6)
    plt.scatter(one[y == 6], two[y == 6], alpha=0.9, color=color7)
    plt.scatter(one[y == 7], two[y == 7], alpha=0.9, color=color8)
    plt.scatter(one[y == 8], two[y == 8], alpha=0.9, color=color9)
    plt.scatter(one[y == 9], two[y == 9], alpha=0.9, color=color10)
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.title('t-SNE of first 50 PCA components')
    plt.xlabel('t-SNE component one')
    plt.ylabel('t-SNE component two')
    plt.savefig(fname=path)
    plt.clf()
