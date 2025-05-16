# Detecção de Transações Fraudulentas com Machine Learning
Este projeto tem como objetivo construir um pipeline completo de ciência de dados para identificar transações financeiras fraudulentas. Desde a análise exploratória até a modelagem e interpretação dos resultados, todo o processo foi estruturado em Python com foco em clareza, desempenho e interpretabilidade.

### Etapas Realizadas
1 - Entendimento do Problema
- O objetivo é detectar fraudes em transações bancárias, minimizando falsos positivos.

2 - Análise Exploratória de Dados (EDA)
- Análise da distribuição de variáveis
- Detecção de outliers
- Visualização por categoria e hora da transação
- Correlações e padrões relevantes

3 - Pré-processamento
- Conversão de variáveis temporais
- Codificação de variáveis categóricas
- Padronização de dados numéricos
- Separação entre treino e teste

4 - Treinamento e Validação de Modelos
- Avaliação de algoritmos como Decision Tree e Random Forest
- Otimização de hiperparâmetros com GridSearchCV
- Avaliação com métricas como AUC, Recall, Precision, F1-score

5 - Feature Importance
- Redução de dimensionalidade com base nas 15 features mais relevantes

### Principais Bibliotecas Usadas
- pandas, numpy, matplotlib, seaborn
- scikit-learn para modelagem
- shap para explicabilidade
- eda_utils.py (funções personalizadas de visualização e avaliação)

### Resultados
O modelo final apresentou:
- Excelente capacidade de detectar transações fraudulentas
- Baixo índice de falsos positivos
- Pipeline reaproveitável para ambientes reais
