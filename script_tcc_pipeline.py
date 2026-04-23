#%%
# =========================================================
# 1. IMPORTS
# =========================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt

#%%
# =========================================================
# 2. CARREGAMENTO E TRATAMENTO
# =========================================================
df = pd.read_csv('df_dados.csv')

cols_datas = ['data_evento', 'entrega_prometida', 'entrega_realizada']
df[cols_datas] = df[cols_datas].apply(pd.to_datetime, errors='coerce')

df = df[df['nome_hub'].notna()]

#%%
# =========================================================
# 3. VARIÁVEIS
# =========================================================
y = df['dias_dif_promessa_entrega']

vars_quant = [
    'hora_evento', 'dia_semana_evento', 'dia_mes_evento',
    'dia_mes_entrega_prometida', 'qtde_total_ocorr_transportador',
    'qtde_total_ocorr_cliente', 'qtde_total_ocorr_taxas',
    'qtde_total_ocorr_causas_naturais', 'qtde_total_ocorr_outros',
    'saldo_delta_t', 'dias_para_entrega_prometida',
    'media_dias_dif_promessa_entrega'
]

vars_cat = [
    'hora_evento_range', 'nome_hub', 'tipo_evento',
    'cep_destino_range', 'armazem_origem', 'regiao_destino',
    'uf_destino', 'area_destino', 'cidade_destino',
    'transportador', 'canal_vendas',
    'divisao_produto', 'categoria_produto'
]

X = df[vars_quant + vars_cat]

#%%
# =========================================================
# 4. SPLIT 
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#%%
# =========================================================
# 5. TARGET ENCODING COM K-FOLD
# =========================================================
def target_encoding_kfold(X, y, cols, n_splits=5):
    X_new = X.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    global_mean = y.mean()

    for col in cols:
        X_new[col] = np.nan

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            means = y_tr.groupby(X_tr[col]).mean()
            X_new.iloc[val_idx, X.columns.get_loc(col)] = X_val[col].map(means)

        X_new[col].fillna(global_mean, inplace=True)

    return X_new


# Treino com KFold
X_train_te = target_encoding_kfold(X_train, y_train, vars_cat)

# Teste com média do treino
X_test_te = X_test.copy()
for col in vars_cat:
    means = y_train.groupby(X_train[col]).mean()
    X_test_te[col] = X_test[col].map(means)

X_test_te[vars_cat] = X_test_te[vars_cat].fillna(y_train.mean())

#%%
# =========================================================
# 6. PADRONIZAÇÃO
# =========================================================
scaler = StandardScaler()

X_train_te.loc[:, vars_quant] = scaler.fit_transform(X_train_te[vars_quant])
X_test_te.loc[:, vars_quant] = scaler.transform(X_test_te[vars_quant])

#%%
# =========================================================
# 7. MODELOS
# =========================================================
modelos_reg = {
    'Linear': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

modelos_class = {
    'Logistic': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}


# Dados estruturados
dados = [
    ["LinearRegression", "Regressão", "Modelo linear simples que assume relação direta entre variáveis."],
    ["RandomForestRegressor", "Regressão", "Ensemble de árvores que captura padrões não lineares e reduz overfitting."],
    ["GradientBoostingRegressor", "Regressão", "Modelo sequencial que corrige erros anteriores, alta precisão."],
    ["SVR", "Regressão", "Regressão baseada em margens, sensível à escala dos dados."],
    ["KNNRegressor", "Regressão", "    , simples porém limitado."],
    
    ["LogisticRegression", "Classificação", "Modelo linear para classificação baseado em probabilidade."],
    ["RandomForestClassifier", "Classificação", "Ensemble robusto de árvores, ótimo para dados tabulares."],
    ["GradientBoostingClassifier", "Classificação", "Modelo sequencial com alta capacidade preditiva."],
    ["SVC", "Classificação", "Classificador baseado em hiperplanos, eficiente em alta dimensionalidade."],
    ["KNNClassifier", "Classificação", "Classificação baseada na maioria dos vizinhos mais próximos."]
]

df_modelos = pd.DataFrame(dados, columns=["Modelo", "Tipo", "Descrição"])
df_modelos = df_modelos.sort_values(by="Modelo")

fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('off')

tabela = ax.table(
    cellText=df_modelos.values,
    colLabels=df_modelos.columns,
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.auto_set_column_width(col=list(range(3)))

for (row, col), cell in tabela.get_celld().items():
    cell.get_text().set_ha('left')  # horizontal alignment
    cell.get_text().set_va('center')  # vertical alignment

for i in range(len(df_modelos) + 1):
    for j in range(3):
        tabela[(i, j)].set_height(0.08)

plt.title("Modelos Utilizados na Análise", fontsize=14)
plt.tight_layout()
plt.show()

#%%
# =========================================================
# 8. FUNÇÃO DE MÉTRICAS DE NEGÓCIO
# =========================================================
def metricas_negocio(y_true, y_pred):
    y_pred_int = np.round(y_pred).astype(int)
    y_pred_int = np.clip(y_pred_int, y_true.min(), y_true.max())

    acc_exato = np.mean(y_pred_int == y_true)
    acc_1 = np.mean(np.abs(y_pred_int - y_true) <= 1)
    acc_2 = np.mean(np.abs(y_pred_int - y_true) <= 2)

    return acc_exato, acc_1, acc_2

#%%
# =========================================================
# 9. AVALIAÇÃO
# =========================================================
resultados = []

# -------- REGRESSÃO --------
for nome, modelo in modelos_reg.items():
    modelo.fit(X_train_te, y_train)
    y_pred = modelo.predict(X_test_te)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    acc_exato, acc_1, acc_2 = metricas_negocio(y_test.values, y_pred)

    resultados.append([
        nome + "_Reg", mae, rmse, r2,
        acc_exato * 100, acc_1 * 100, acc_2 * 100
    ])


# -------- CLASSIFICAÇÃO --------
y_train_c = y_train.astype(int)
y_test_c = y_test.astype(int)

for nome, modelo in modelos_class.items():
    modelo.fit(X_train_te, y_train_c)
    y_pred = modelo.predict(X_test_te)

    mae = mean_absolute_error(y_test_c, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_c, y_pred))

    acc_exato, acc_1, acc_2 = metricas_negocio(y_test_c.values, y_pred)

    resultados.append([
        nome + "_Class", mae, rmse, np.nan,
        acc_exato * 100, acc_1 * 100, acc_2 * 100
    ])

#%%
# =========================================================
# 10. RESULTADO FINAL
# =========================================================
df_resultado = pd.DataFrame(resultados, columns=[
    'Modelo', 'MAE', 'RMSE', 'R2',
    'Acerto Exato (%)', 'Acerto ±1 dia (%)', 'Acerto ±2 dias (%)'
])

df_resultado = df_resultado.sort_values('MAE')

print("\nRESULTADO FINAL")
print(df_resultado)



df_plot = df_resultado.copy()
df_plot = df_plot.sort_values(by='Acerto Exato (%)', ascending=False).reset_index(drop=True)

df_plot.insert(0, 'Rank', df_plot.index + 1)

df_plot['MAE'] = df_plot['MAE'].round(3)
df_plot['RMSE'] = df_plot['RMSE'].round(3)
df_plot['R2'] = df_plot['R2'].round(3)

df_plot['Acerto Exato (%)'] = df_plot['Acerto Exato (%)'].round(2)
df_plot['Acerto ±1 dia (%)'] = df_plot['Acerto ±1 dia (%)'].round(2)
df_plot['Acerto ±2 dias (%)'] = df_plot['Acerto ±2 dias (%)'].round(2)

fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('off')

tabela = ax.table(
    cellText=df_plot.values,
    colLabels=df_plot.columns,
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.auto_set_column_width(col=list(range(len(df_plot.columns))))

for (row, col), cell in tabela.get_celld().items():
    cell.get_text().set_ha('left')
    cell.get_text().set_va('center')

for i in range(len(df_plot) + 1):
    for j in range(len(df_plot.columns)):
        tabela[(i, j)].set_height(0.08)

plt.title("Ranking de Modelos - Acerto Exato (%)", fontsize=14)
plt.tight_layout()
plt.show()

#%%
# =========================================================
# 11. FEATURE IMPORTANCE + SELEÇÃO AUTOMÁTICA (RF CLASS)
# =========================================================
from sklearn.inspection import permutation_importance

# =========================================================
# 11.1. TREINAR RANDOM FOREST CLASSIFIER
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

y_train_c = y_train.astype(int)
y_test_c = y_test.astype(int)

rf_model.fit(X_train_te, y_train_c)

# =========================================================
# 11.2. PERMUTATION IMPORTANCE
# =========================================================
perm = permutation_importance(
    rf_model,
    X_test_te,
    y_test_c,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

feat_imp = pd.DataFrame({
    'feature': X_train_te.columns,
    'importance': perm.importances_mean
}).sort_values(by='importance', ascending=False)

print("\nTop 20 features:")
print(feat_imp.head(20))

# =========================================================
# 11.3. DEFINIR LIMIAR AUTOMÁTICO
# =========================================================
# manter features acima da mediana ou > 0

threshold = max(0, feat_imp['importance'].median())

features_selecionadas = feat_imp[
    feat_imp['importance'] > threshold
]['feature'].tolist()

features_removidas = feat_imp[
    feat_imp['importance'] <= threshold
]['feature'].tolist()

print("\nFeatures selecionadas:", len(features_selecionadas))
print(features_selecionadas)

print("\nFeatures removidas:", len(features_removidas))
print(features_removidas)


feat_imp_plot = feat_imp.sort_values(by='importance', ascending=True).reset_index(drop=True)

mediana = feat_imp_plot['importance'].median()

feat_imp_plot['selecionada'] = feat_imp_plot['importance'] > mediana
cut_index = feat_imp_plot[feat_imp_plot['selecionada'] == False].shape[0]

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.barh(feat_imp_plot['feature'], feat_imp_plot['importance'])

ax.axhline(y=cut_index - 0.5, linestyle='--', linewidth=2)

ax.text(
    x=feat_imp_plot['importance'].max() * 0.95,  # quase no limite direito
    y=cut_index - 0.5 + 0.3,
    s=f"Mediana {mediana:.4f}",
    fontsize=10,
    ha='right'  # alinha o texto pela direita
)

for i, v in enumerate(feat_imp_plot['importance']):
    ax.text(
        x=v + (feat_imp_plot['importance'].max() * 0.01),
        y=i,
        s=f"{v:.3f}",
        va='center',
        fontsize=9
    )

ax.text(
    x=feat_imp_plot['importance'].max() * 0.7,
    y=cut_index + (len(feat_imp_plot) - cut_index) * 0.5,
    s="Selecionadas",
    fontsize=11
)

ax.text(
    x=feat_imp_plot['importance'].max() * 0.7,
    y=cut_index * 0.3,
    s="Rejeitadas",
    fontsize=11
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("Importância (Permutation Importance)")
ax.set_ylabel("Features")

plt.tight_layout()
plt.show()

# =========================================================
# 12. REDUZIR BASE
# =========================================================
X_train_sel = X_train_te[features_selecionadas]
X_test_sel = X_test_te[features_selecionadas]

# =========================================================
# 13. GRID SEARCH + CROSS VALIDATION (RF CLASS)
# =========================================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5]
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# Treinar com features selecionadas
grid_search.fit(X_train_sel, y_train_c)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

print("\nMelhores hiperparâmetros:")
print(grid_search.best_params_)

# =========================================================
# 14. PREDIÇÃO COM MELHOR MODELO
# =========================================================
y_pred_sel = best_model.predict(X_test_sel)

# =========================================================
# 15. MÉTRICAS
# =========================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_c, y_pred_sel)
rmse = np.sqrt(mean_squared_error(y_test_c, y_pred_sel))

def metricas_negocio(y_true, y_pred):
    acc_exato = np.mean(y_pred == y_true)
    acc_1 = np.mean(np.abs(y_pred - y_true) <= 1)
    acc_2 = np.mean(np.abs(y_pred - y_true) <= 2)
    return acc_exato, acc_1, acc_2

acc_exato, acc_1, acc_2 = metricas_negocio(y_test_c, y_pred_sel)

print("\nRESULTADO FINAL (RF + GRID SEARCH + FEATURE SELECTION)")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Acerto exato: {acc_exato*100:.2f}%")
print(f"Acerto ±1 dia: {acc_1*100:.2f}%")
print(f"Acerto ±2 dias: {acc_2*100:.2f}%")

# =========================================================
# 16. COMPARAÇÃO (ANTES vs DEPOIS)
# =========================================================

# Resultado anterior do RF Class
rf_before = df_resultado[df_resultado['Modelo'] == 'RandomForest_Class']

comparacao = pd.DataFrame({
    'Métrica': ['MAE', 'RMSE', 'Acerto Exato', '±1 dia', '±2 dias'],
    'Antes': [
        rf_before['MAE'].values[0],
        rf_before['RMSE'].values[0],
        rf_before['Acerto Exato (%)'].values[0],
        rf_before['Acerto ±1 dia (%)'].values[0],
        rf_before['Acerto ±2 dias (%)'].values[0]
    ],
    'Depois': [
        mae,
        rmse,
        acc_exato * 100,
        acc_1 * 100,
        acc_2 * 100
    ]
})

print("\nCOMPARAÇÃO REAL (ANTES vs DEPOIS)")
print(comparacao)

# =========================================================
# 17. IMPORTÂNCIA FINAL (MODELO OTIMIZADO)
# =========================================================
from sklearn.inspection import permutation_importance

perm_final = permutation_importance(
    best_model,
    X_test_sel,
    y_test_c,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

feat_imp_final = pd.DataFrame({
    'feature': X_train_sel.columns,
    'importance': perm_final.importances_mean
}).sort_values(by='importance', ascending=False)

print("\nTop 15 features finais:")
print(feat_imp_final.head(15))

# =========================================================
# 18. PLOT FINAL
# =========================================================
top_n = 15
top_features = feat_imp_final.head(top_n)


fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(top_features['feature'], top_features['importance'])

ax.invert_yaxis()

x_max = top_features['importance'].max()
for i, v in enumerate(top_features['importance']):
    ax.text(
        x=v + x_max * 0.01,  # pequeno offset à direita
        y=i,
        s=f"{v:.3f}",
        va='center',
        fontsize=9
    )

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

ax.set_xlabel('Importância')
ax.set_ylabel('Features')

plt.tight_layout()
plt.show()