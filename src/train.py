"""
Modelamiento predictivo y explicabilidad (Componente B).

Compara Regresión Logística (línea base) contra Random Forest y XGBoost,
reporta Recall/Precision/F1/AUC-ROC, valida con k-fold, selecciona el mejor
modelo basado en árboles (compatible con SHAP TreeExplainer) y serializa
pipeline + explicador para su consumo por app/api.py y app/main_dashboard.py.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import data_pipeline, decision_engine  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = Path("models")
PROCESSED_CSV = data_pipeline.PROCESSED_DIR / "processed_appointments.csv"

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


# --------------------------------------------------------------------------
# Datos y preprocesamiento
# --------------------------------------------------------------------------

def load_processed_data() -> pd.DataFrame:
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
    else:
        print("[train] processed_appointments.csv no encontrado; ejecutando data_pipeline ...")
        df = data_pipeline.run(save_processed_csv=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), data_pipeline.FEATURE_COLUMNS_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             data_pipeline.FEATURE_COLUMNS_CATEGORICAL),
        ]
    )


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols = data_pipeline.FEATURE_COLUMNS_NUMERIC + data_pipeline.FEATURE_COLUMNS_CATEGORICAL
    X = df[cols].copy()
    y = df[data_pipeline.TARGET_COLUMN].copy()
    return X, y


# --------------------------------------------------------------------------
# Entrenamiento y comparación de modelos
# --------------------------------------------------------------------------

def _candidate_models(scale_pos_weight: float) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    else:  # pragma: no cover - fallback si xgboost no está instalado
        print("[train] xgboost no disponible; se usa GradientBoosting como sustituto.")
        models["GradientBoosting(fallback XGBoost)"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, random_state=42
        )
    return models


def evaluate_model(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo": name,
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    return metrics


def train_and_select_best(df: pd.DataFrame) -> dict:
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = _candidate_models(scale_pos_weight)

    results = []
    fitted_pipelines = {}
    for name, estimator in models.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        print(f"[train] Entrenando y validando: {name} ...")
        metrics = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(metrics)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    print("\n=== Comparación de modelos (conjunto de prueba) ===")
    print(results_df.to_string(index=False))

    tree_based = [r["modelo"] for r in results if r["modelo"] != "LogisticRegression"]
    tree_results = results_df[results_df["modelo"].isin(tree_based)]
    best_name = tree_results.sort_values("auc_roc", ascending=False).iloc[0]["modelo"]
    best_pipeline = fitted_pipelines[best_name]
    best_metrics = next(r for r in results if r["modelo"] == best_name)

    print(f"\n[train] Modelo seleccionado (compatible con SHAP TreeExplainer): {best_name}")
    print(f"[train] AUC-ROC = {best_metrics['auc_roc']} "
          f"(meta RNF: > 0.70 | referencia trivial: 0.50)")

    return {
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "best_metrics": best_metrics,
        "all_results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# --------------------------------------------------------------------------
# Explicabilidad SHAP (RNF-01, RF-04)
# --------------------------------------------------------------------------

def build_explainer(best_pipeline: Pipeline, X_background: pd.DataFrame):
    import shap

    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    background = preprocessor.transform(X_background.sample(
        n=min(200, len(X_background)), random_state=42
    ))
    explainer = shap.TreeExplainer(classifier, data=background, model_output="probability")
    return explainer


def _extract_class1_shap(explainer, X_transformed: np.ndarray) -> np.ndarray:
    """Normaliza la salida de SHAP entre RandomForest (3D, por clase) y
    XGBoost/GradientBoosting (2D) a un arreglo (n_samples, n_features) para
    la clase positiva (no-show=1)."""
    explanation = explainer(X_transformed)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def get_case_explanation(shap_row: np.ndarray, feature_names: list[str], top_n: int = 3) -> list[dict]:
    """RF-04: devuelve las top_n variables con mayor contribución positiva
    hacia la inasistencia, dado un vector de shap values ya calculado."""
    pairs = list(zip(feature_names, shap_row))
    pairs.sort(key=lambda p: p[1], reverse=True)
    top = pairs[:top_n]
    return [
        {"feature": _clean_feature_name(name), "shap_value": round(float(val), 4),
         "empuja_hacia": "inasistencia" if val > 0 else "asistencia"}
        for name, val in top
    ]


def _clean_feature_name(raw_name: str) -> str:
    return raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name


# --------------------------------------------------------------------------
# Serialización de artefactos
# --------------------------------------------------------------------------

def save_artifacts(training_result: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline: Pipeline = training_result["best_pipeline"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]

    explainer = build_explainer(best_pipeline, training_result["X_train"])
    feature_names = list(preprocessor.get_feature_names_out())

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(classifier, MODELS_DIR / "trained_model.joblib")
    joblib.dump(explainer, MODELS_DIR / "explainer.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    joblib.dump({
        "model_name": training_result["best_name"],
        "metrics": training_result["best_metrics"],
        "all_results": training_result["all_results"],
        "risk_thresholds": decision_engine.RISK_THRESHOLDS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }, MODELS_DIR / "metadata.joblib")
    print(f"[train] Artefactos guardados en {MODELS_DIR}/")


# --------------------------------------------------------------------------
# Carga e inferencia en tiempo de servicio (usado por app/api.py y dashboard)
# --------------------------------------------------------------------------

_ARTIFACTS_CACHE: dict = {}


def load_artifacts(force_reload: bool = False) -> dict:
    if _ARTIFACTS_CACHE and not force_reload:
        return _ARTIFACTS_CACHE
    _ARTIFACTS_CACHE["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.joblib")
    _ARTIFACTS_CACHE["model"] = joblib.load(MODELS_DIR / "trained_model.joblib")
    _ARTIFACTS_CACHE["explainer"] = joblib.load(MODELS_DIR / "explainer.joblib")
    _ARTIFACTS_CACHE["feature_names"] = joblib.load(MODELS_DIR / "feature_names.joblib")
    _ARTIFACTS_CACHE["metadata"] = joblib.load(MODELS_DIR / "metadata.joblib")
    return _ARTIFACTS_CACHE


def artifacts_available() -> bool:
    required = ["preprocessor.joblib", "trained_model.joblib", "explainer.joblib",
                "feature_names.joblib", "metadata.joblib"]
    return all((MODELS_DIR / f).exists() for f in required)


def predict_case(patient_features: dict, top_n: int = 3) -> dict:
    """RF-02/RF-03/RF-04: probabilidad, nivel de riesgo y top-N variables SHAP
    para un caso individual (usado por POST /api/v1/predict-risk y la UI)."""
    artifacts = load_artifacts()
    preprocessor = artifacts["preprocessor"]
    model = artifacts["model"]
    explainer = artifacts["explainer"]
    feature_names = artifacts["feature_names"]

    cols = data_pipeline.FEATURE_COLUMNS_NUMERIC + data_pipeline.FEATURE_COLUMNS_CATEGORICAL
    row = {c: patient_features.get(c) for c in cols}
    X = pd.DataFrame([row])

    X_transformed = preprocessor.transform(X)
    probability = float(model.predict_proba(X_transformed)[0, 1])
    risk_level = decision_engine.classify_risk(probability, artifacts["metadata"]["risk_thresholds"])

    shap_values = _extract_class1_shap(explainer, X_transformed)
    top_features = get_case_explanation(shap_values[0], feature_names, top_n=top_n)

    return {
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "top_features": top_features,
        "model_name": artifacts["metadata"]["model_name"],
    }


def main() -> None:
    df = load_processed_data()
    training_result = train_and_select_best(df)
    save_artifacts(training_result)


if __name__ == "__main__":
    main()
