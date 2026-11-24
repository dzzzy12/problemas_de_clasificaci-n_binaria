"""Clasificación KNN de satisfacción de clientes con datos sintéticos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEMILLA = 42
NUMERICAS = [
    "tiempo_respuesta_min",
    "calidad_atencion",
    "interacciones",
    "antiguedad_cliente_meses",
]
CATEGORICAS = ["resolucion_primer_contacto", "expectativa_cumplida", "canal"]
CLASES = ["No satisfecho", "Satisfecho"]
_modelo_entrenado: Pipeline | None = None


def generar_dataset(n: int = 200, semilla: int = SEMILLA) -> pd.DataFrame:
    """Genera un dataset reproducible y balanceado de satisfacción."""
    if n != 200:
        raise ValueError("El dataset de este ejercicio debe contener 200 registros.")

    rng = np.random.default_rng(semilla)
    tiempo = rng.integers(1, 61, size=n)
    calidad = rng.integers(1, 6, size=n)
    interacciones = rng.integers(1, 6, size=n)
    antiguedad = rng.integers(1, 121, size=n)
    resolucion = rng.choice(["Sí", "No"], size=n, p=[0.72, 0.28])
    expectativa = rng.choice(["Sí", "No"], size=n, p=[0.68, 0.32])
    canal = rng.choice(
        ["Chat", "Correo", "Teléfono", "Presencial"],
        size=n,
        p=[0.35, 0.25, 0.30, 0.10],
    )

    puntaje = (
        1.10 * (calidad - 3)
        + 1.20 * (resolucion == "Sí")
        + 1.35 * (expectativa == "Sí")
        - 0.035 * tiempo
        - 0.32 * (interacciones - 1)
        + rng.normal(0, 0.9, size=n)
    )
    satisfecho = np.zeros(n, dtype=bool)
    satisfecho[np.argsort(puntaje, kind="stable")[n // 2 :]] = True

    return pd.DataFrame(
        {
            "tiempo_respuesta_min": tiempo,
            "calidad_atencion": calidad,
            "interacciones": interacciones,
            "antiguedad_cliente_meses": antiguedad,
            "resolucion_primer_contacto": resolucion,
            "expectativa_cumplida": expectativa,
            "canal": canal,
            "satisfaccion": np.where(satisfecho, CLASES[1], CLASES[0]),
        }
    )


def crear_pipeline() -> Pipeline:
    """Construye el preprocesamiento y el clasificador KNN."""
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), NUMERICAS),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", KNeighborsClassifier()),
        ]
    )


def entrenar_modelo() -> tuple[GridSearchCV, pd.DataFrame, pd.Series]:
    """Busca k en entrenamiento y devuelve el modelo y el conjunto de prueba."""
    datos = generar_dataset()
    X = datos[NUMERICAS + CATEGORICAS]
    y = datos["satisfaccion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMILLA
    )

    busqueda = GridSearchCV(
        estimator=crear_pipeline(),
        param_grid={"clasificador__n_neighbors": list(range(3, 16, 2))},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    return busqueda, X_test, y_test


def predecir_nuevo_caso(caso: dict[str, Any]) -> dict[str, str | float]:
    """Predice satisfacción y devuelve la probabilidad de estar satisfecho."""
    global _modelo_entrenado

    faltantes = set(NUMERICAS + CATEGORICAS) - caso.keys()
    sobrantes = caso.keys() - set(NUMERICAS + CATEGORICAS)
    if faltantes or sobrantes:
        raise ValueError(
            f"Campos faltantes: {sorted(faltantes)}; campos desconocidos: {sorted(sobrantes)}"
        )

    if _modelo_entrenado is None:
        busqueda, _, _ = entrenar_modelo()
        _modelo_entrenado = busqueda.best_estimator_

    entrada = pd.DataFrame([caso], columns=NUMERICAS + CATEGORICAS)
    prediccion = str(_modelo_entrenado.predict(entrada)[0])
    indice_satisfecho = list(_modelo_entrenado.classes_).index(CLASES[1])
    probabilidad = float(_modelo_entrenado.predict_proba(entrada)[0][indice_satisfecho])
    return {"prediccion": prediccion, "probabilidad": probabilidad}


def main() -> None:
    """Entrena el modelo e imprime sus métricas de evaluación."""
    global _modelo_entrenado

    busqueda, X_test, y_test = entrenar_modelo()
    modelo = busqueda.best_estimator_
    _modelo_entrenado = modelo
    predicciones = modelo.predict(X_test)

    print("=" * 60)
    print("MODELO KNN: SATISFACCIÓN DEL CLIENTE")
    print("=" * 60)
    print(f"Mejor k: {busqueda.best_params_['clasificador__n_neighbors']}")
    print(f"Accuracy: {accuracy_score(y_test, predicciones):.3f}")
    print("Matriz de confusión (filas: real, columnas: predicción):")
    print(confusion_matrix(y_test, predicciones, labels=CLASES))
    print("Reporte de clasificación:")
    print(
        classification_report(
            y_test, predicciones, labels=CLASES, target_names=CLASES, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
