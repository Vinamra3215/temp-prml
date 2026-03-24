"""UMAP interactive scatter plot."""
import numpy as np


def plot_umap_interactive(X, y, class_names, title="UMAP Embedding", save_path=None):
    """Create interactive UMAP scatter plot with Plotly."""
    import umap
    import plotly.express as px

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)

    labels = [class_names[i] if i < len(class_names) else str(i) for i in y]
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1], color=labels,
        title=title, opacity=0.6, width=900, height=700,
    )
    if save_path:
        fig.write_html(save_path)
    return fig
