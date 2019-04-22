from sklearn.manifold import TSNE
import matplot.pylab as plt
def plot_tsne(arr):
    """
        arr = [np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
           np.array([[0, 0, 0], [0, 1, 3], [1, 4, 1], [1, 1, 1]])]
    :param arr:
    :return:
    """
    for i,X in enumerate(arr):
        X_embedded = TSNE(n_components=2).fit_transform(X)
        plt.plot(X,label=i)
    plt.save_figure('tsne.png')
