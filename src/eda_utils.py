import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_predict

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

def avaliar_modelo(model_name, results, X_train, y_train):
    """
    Avalia modelo com matriz de confusão, classification report,
    curvas ROC e PR, distribuição de erros e feature importances (se disponível).
    """
    print(f"\nAvaliando modelo: {model_name}")
    best_model = results[model_name]['best_estimator_']
    
    # Previsões com cross_val_predict
    y_pred = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict')

    # Matriz de confusão
    cm = confusion_matrix(y_train, y_pred)
    cr = classification_report(y_train, y_pred)

    print("Classification Report:")
    print(cr)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

    # Curva ROC (se binária)
    if len(np.unique(y_train)) == 2:
        y_proba = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color=palette[0], label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()

        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_train, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, color=palette[1], label=f'AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.legend()
        plt.show()

    # Distribuição de Erros
    erro = y_pred != y_train
    plt.figure(figsize=(6, 4))
    sns.histplot(erro, bins=2, palette=palette)
    plt.title('Distribuição de Erros (False = acerto, True = erro)')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.show()

    # Feature Importances
    try:
        classifier = best_model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_

            preprocessor = best_model.named_steps['preprocessor']
            transformed_cols = []

            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':
                    transformed_cols.extend(cols)
                elif name in ['ordinal_enc', 'target_enc']:
                    transformed_cols.extend(cols)
                elif name == 'onehot':
                    onehot = transformer
                    onehot_feature_names = onehot.get_feature_names_out(cols)
                    transformed_cols.extend(onehot_feature_names)

            # Caso use PCA, redefinir nomes das features
            if 'pca' in best_model.named_steps:
                n_components = best_model.named_steps['pca'].n_components_
                transformed_cols = [f'PC{i+1}' for i in range(n_components)]

            feat_imp = pd.Series(importances, index=transformed_cols)
            feat_imp = feat_imp.sort_values(ascending=False).head(20)

            plt.figure(figsize=(8, 5))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette=palette)
            plt.title('Feature Importances')
            plt.xlabel('Importância')
            plt.ylabel('Features')
            plt.show()
        else:
            print("Modelo não possui feature_importances_.")
    except Exception as e:
        print(f"Não foi possível plotar feature importances: {e}")