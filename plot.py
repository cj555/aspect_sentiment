def plot_tsne(arr):
    for X in arr:
        X_embedded = TSNE(n_components=2).fit_transform(X)
