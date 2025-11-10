
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 00:45:45 2025

@author: Equipo DSA
"""

# ===================== Ajustes mínimos solicitados =====================
import os
import mlflow
import mlflow.sklearn

# Tracking dinámico: variable de entorno o localhost si no está definida
mlflow.set_tracking_uri("http://54.90.215.224:8050")
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050"))

# DATA_PATH opcional por variable de entorno (mantiene tu default)
DATA_PATH = os.getenv("DATA_PATH", "base_balanceada_mod_final.csv")
# ======================================================================

# Importe la base final y divídala en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)

# Variables a excluir del modelo
DROP_FROM_MODEL = ['NIU', 'ALTITUD', 'LONGITUD', 'LATITUD']
TARGET = 'COMPENSADO'

# y (target) y X (features)
y = df[TARGET].astype(int)
X = df.drop(columns=DROP_FROM_MODEL + [TARGET], errors='ignore')

# Split (igual estructura que el ejemplo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

# Importe MLflow para registrar los experimentos, el clasificador LDA y la métrica AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

# Preprocesamiento mínimo necesario:
# LDA requiere datos numéricos -> OneHot para categóricas y escalado para numéricas
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identificar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# Registre el experimento
experiment = mlflow.set_experiment("sklearn-compensado-lda")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento adicional.
# MLflow crea el experimento si no existe y guarda parámetros/métricas/artefactos definidos.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Defina los parámetros del modelo
    solver = "svd"        # ('svd', 'lsqr', 'eigen')
    shrinkage = None      # Solo aplica si solver es 'lsqr' o 'eigen'

    # Cree el modelo con los parámetros definidos y entrénelo
    lda = LDA(solver=solver, shrinkage=shrinkage)
    model = Pipeline([
        ("prep", preproc),
        ("clf", lda),
    ])
    model.fit(X_train, y_train)

    # Realice predicciones de prueba (probabilidades para AUC)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Registre parámetros y metadatos básicos
    mlflow.log_param("solver", solver)
    mlflow.log_param("shrinkage", shrinkage)
    mlflow.log_param("test_split_random_state", 42)
    mlflow.log_param("data_path", DATA_PATH)

    # Métrica AUC
    auc = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("auc", float(auc))
    print(auc)

    # ---- Ajuste nuevo para eliminar warnings (signature + input_example, API nueva) ----
    from mlflow.models import infer_signature
    signature = infer_signature(X_train.head(5), model.predict_proba(X_train.head(5))[:, 1])

    mlflow.sklearn.log_model(
        sk_model=model,
        name="lda-model",          # API nueva: usar 'name' en lugar de 'artifact_path'
        signature=signature,
        input_example=X_train.head(2)
    )
    # ---------------------------------------------------------------------

