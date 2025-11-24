import prince
def run_pca(dataset, components):

    pca_model = prince.PCA(
        n_components=components,
        n_iter=3,
        rescale_with_mean=True,
        rescale_with_std=True,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )

    pca_model = pca_model.fit(dataset)

    return pca_model