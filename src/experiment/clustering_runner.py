"""
Clustering experiment runner.
"""
from src.data.cache import load_features
from src.clustering.kmeans import run_kmeans, evaluate_clustering, elbow_plot
from src.clustering.agglomerative import run_agglomerative, plot_dendrogram
from src.clustering.gmm import run_gmm, bic_aic_sweep


def run_clustering_experiments(feature_type, cache_dir="data/cache", plots_dir="results/plots"):
    """Run all clustering experiments for a given feature type."""
    import os
    os.makedirs(plots_dir, exist_ok=True)

    X_train, y_train = load_features(cache_dir, feature_type, "train")
    print(f"\nClustering on {feature_type} features: {X_train.shape}")

    # KMeans
    _, km_labels = run_kmeans(X_train, n_clusters=20)
    km_metrics = evaluate_clustering(X_train, km_labels, y_train)
    print(f"KMeans: {km_metrics}")

    # Agglomerative
    _, agg_labels = run_agglomerative(X_train, n_clusters=20)
    agg_metrics = evaluate_clustering(X_train, agg_labels, y_train)
    print(f"Agglomerative: {agg_metrics}")

    # GMM
    _, gmm_labels = run_gmm(X_train, n_components=20)
    gmm_metrics = evaluate_clustering(X_train, gmm_labels, y_train)
    print(f"GMM: {gmm_metrics}")

    # Plots
    elbow_plot(X_train, save_path=os.path.join(plots_dir, f"elbow_{feature_type}.png"))
    plot_dendrogram(X_train, save_path=os.path.join(plots_dir, f"dendrogram_{feature_type}.png"))
    bic_aic_sweep(X_train, save_path=os.path.join(plots_dir, f"bic_aic_{feature_type}.png"))

    return {"kmeans": km_metrics, "agglomerative": agg_metrics, "gmm": gmm_metrics}
