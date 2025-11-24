
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
DATA_PATH = os.getenv("DATA_PATH", "base_balanceada_mod_final_dash.csv")
# ======================================================================

# Importe la base final y divídala en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)

# Variables a excluir del modelo
DROP_FROM_MODEL = ['NIU', 'ALTITUD', 'LONGITUD', 'LATITUD', 'PERIODO']
TARGET = 'COMPENSADO'

# y (target) y X (features)
y = df[TARGET].astype(int)
X = df.drop(columns=DROP_FROM_MODEL + [TARGET], errors='ignore')

# Split (igual estructura que el ejemplo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

# Importe MLflow para registrar los experimentos, el clasificador Random Forest y la métrica AUC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Preprocesamiento mínimo necesario:
# (mantiene tu esquema original de escalado+OneHot)
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

# Registre el experimento (nombre ajustado)
experiment = mlflow.set_experiment("sklearn-compensado-randomforest")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento adicional.
# MLflow crea el experimento si no existe y guarda parámetros/métricas/artefactos definidos.
with mlflow.start_run(experiment_id=experiment.experiment_id):

    # ─────────────────────────────────────────────
    # Defina los parámetros del modelo
    # (misma estructura del código original)
    # ─────────────────────────────────────────────
    n_estimators     = 300
    max_depth        = 20
    min_samples_split = 0.01
    min_samples_leaf  = 0.005
    class_weight     = "balanced"

    # Cree el modelo con los parámetros definidos y entrénelo
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("prep", preproc),
        ("clf", rf),
    ])

    model.fit(X_train, y_train)

    # Realice predicciones de prueba (probabilidades para AUC)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Registre parámetros y metadatos básicos
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("class_weight", class_weight)
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
        name="rf-model",     # <-- cambiado
        signature=signature,
        input_example=X_train.head(2)
    )
    # ---------------------------------------------------------------------