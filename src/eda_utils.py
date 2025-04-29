import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

palette = sns.color_palette(['#023047', '#e85d04', '#0077b6', '#ff8200', '#0096c7', '#ff9c33'])

def plot_graph(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,
               outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
               scatter=False, scatter_y=None, color='#023047', figsize=(24, 12)):
    """
    Plota diversos tipos de gráficos para análise exploratória.
    """

    num_features = len(features)
    num_cols = 3
    num_rows = num_features // num_cols + int(num_features % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, feature in enumerate(features):
        ax = axes[i]

        # Barchart
        if barplot:
            if mean:
                grouped = data.groupby([feature])[[mean]].mean().reset_index()
                grouped[mean] = round(grouped[mean], 2)
                ax.barh(grouped[feature], grouped[mean], color=color)
                for idx, val in enumerate(grouped[mean]):
                    ax.text(val + text_y, idx, f'{val:.1f}', va='center', fontsize=12)
            else:
                if hue:
                    grouped = data.groupby([feature])[hue].mean().reset_index().rename(columns={hue: 'pct'})
                    grouped['pct'] *= 100
                else:
                    grouped = data[feature].value_counts(normalize=True).reset_index()
                    grouped.columns = [feature, 'pct']
                    grouped['pct'] *= 100

                ax.barh(grouped[feature], grouped['pct'], color=color)
                for idx, val in enumerate(grouped['pct']):
                    ax.text(val + text_y, idx, f'{val:.1f}%', va='center', fontsize=12)

        # Boxplot
        elif outliers and not boxplot:
            sns.boxplot(data=data, x=feature, ax=ax, color=color)

        elif boxplot:
            sns.boxplot(data=data, y=boxplot_x, x=feature, ax=ax, palette=palette, showfliers=outliers)

        # Scatterplot
        elif scatter and scatter_y:
            sns.scatterplot(data=data, x=feature, y=scatter_y, hue=hue, palette=palette if hue else None, ax=ax)

        # Histograma
        elif histplot:
            try:
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='density' if kde else 'proportion', hue=hue, palette=palette if hue else None)
            except Exception:
                sns.histplot(data=data, x=feature, kde=False, ax=ax, color=color, stat='proportion', hue=hue, palette=palette if hue else None)

        ax.set_title(feature, fontsize=14)
        ax.set_xlabel('')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove gráficos não usados
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()