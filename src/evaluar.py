"""Compara el modelo base (linear probing) con el ajustado (ajuste fino completo).

- Exactitud y F1 macro de cada archivo de predicciones.
- Promedio y desviacion estandar si hay varias semillas por modo.
- Matriz de confusion de cada modo (imagen PNG).
- Prueba de McNemar entre el base y cada corrida del ajustado.

Ejemplo:
    python evaluar.py --base resultados/lp_*_predicciones.csv \\
                      --ajustado resultados/ft_*_predicciones.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402

from comun import CARPETA_RESULTADOS, leer_clases  # noqa: E402


def metricas(df):
    return accuracy_score(df["real"], df["prediccion"]), f1_score(df["real"], df["prediccion"], average="macro")


def resumen_modo(nombre, archivos):
    filas = []
    for a in archivos:
        acc, f1 = metricas(pd.read_csv(a))
        filas.append((os.path.basename(a), acc, f1))
        print(f"  {os.path.basename(a):55s} exactitud={acc:.4f}  F1 macro={f1:.4f}")
    accs, f1s = np.array([f[1] for f in filas]), np.array([f[2] for f in filas])
    if len(filas) > 1:
        print(f"  {nombre}: exactitud {accs.mean():.4f} ± {accs.std(ddof=1):.4f} | "
              f"F1 macro {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}  (n={len(filas)})")
    return filas


def matriz(archivo, clases, titulo, salida):
    df = pd.read_csv(archivo)
    cm = confusion_matrix(df["real"], df["prediccion"], labels=clases, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                           colorbar=False, xticks_rotation=45)
    ax.set_title(titulo)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"  Matriz de confusión guardada en {salida}")


def prueba_mcnemar(archivo_base, archivo_ajustado):
    a = pd.read_csv(archivo_base)
    b = pd.read_csv(archivo_ajustado)
    df = a.merge(b, on=["ruta", "real"], suffixes=("_base", "_ajustado"))
    acierta_base = df["prediccion_base"] == df["real"]
    acierta_ajus = df["prediccion_ajustado"] == df["real"]
    b_ = int((acierta_base & ~acierta_ajus).sum())   # base acierta, ajustado falla
    c_ = int((~acierta_base & acierta_ajus).sum())   # base falla, ajustado acierta
    tabla = [[int((acierta_base & acierta_ajus).sum()), b_], [c_, int((~acierta_base & ~acierta_ajus).sum())]]
    exacta = (b_ + c_) < 25
    r = mcnemar(tabla, exact=exacta, correction=True)
    tipo = "exacta (binomial)" if exacta else "chi-cuadrado con corrección"
    print(f"  {os.path.basename(archivo_ajustado)}: b={b_}, c={c_}, estadístico={r.statistic:.3f}, "
          f"p={r.pvalue:.4g} ({tipo}) -> {'significativa' if r.pvalue < 0.05 else 'no significativa'} al 5 %")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", nargs="+", required=True, help="Predicciones de linear probing")
    p.add_argument("--ajustado", nargs="+", required=True, help="Predicciones del ajuste fino completo")
    args = p.parse_args()
    clases = leer_clases()

    print("\nModelo base (linear probing)")
    resumen_modo("Base", args.base)
    print("\nModelo ajustado (ajuste fino completo)")
    resumen_modo("Ajustado", args.ajustado)

    print("\nReporte por material del modelo ajustado (primera corrida)")
    df = pd.read_csv(args.ajustado[0])
    print(classification_report(df["real"], df["prediccion"], labels=clases, digits=3, zero_division=0))

    print("Matrices de confusión")
    matriz(args.base[0], clases, "Linear probing", os.path.join(CARPETA_RESULTADOS, "confusion_base.png"))
    matriz(args.ajustado[0], clases, "Ajuste fino completo", os.path.join(CARPETA_RESULTADOS, "confusion_ajustado.png"))

    print("\nPrueba de McNemar (base vs. cada corrida ajustada)")
    for archivo in args.ajustado:
        prueba_mcnemar(args.base[0], archivo)


if __name__ == "__main__":
    main()
