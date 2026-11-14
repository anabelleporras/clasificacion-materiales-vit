"""Modelo base: linear probing.

1. Pasa todas las imagenes UNA vez por el ViT congelado y guarda sus rasgos.
2. Entrena una regresion logistica multinomial (= capa lineal + softmax +
   entropia cruzada) sobre esos rasgos. El parametro de regularizacion C se
   elige con el conjunto de validacion.
3. Predice el conjunto de prueba y guarda las predicciones.

Ejemplo:
    python linear_probe.py --modelo vit_small_patch16_224
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from comun import (CARPETA_RESULTADOS, DatasetMateriales, crear_modelo, elegir_dispositivo,
                   fijar_semilla, guardar_predicciones, leer_clases, leer_split, transformaciones)


@torch.no_grad()
def extraer_rasgos(modelo, muestras, transformacion, dispositivo, lote, num_workers):
    cargador = DataLoader(DatasetMateriales(muestras, transformacion), batch_size=lote,
                          shuffle=False, num_workers=num_workers)
    rasgos, etiquetas = [], []
    for imagenes, ys in tqdm(cargador, desc="  extrayendo", leave=False):
        rasgos.append(modelo(imagenes.to(dispositivo)).float().cpu().numpy())
        etiquetas.append(ys.numpy())
    return np.concatenate(rasgos), np.concatenate(etiquetas)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", default="vit_small_patch16_224", help="Nombre del modelo en timm")
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--sin-preentrenar", action="store_true", help="Solo para probar el codigo sin descargar pesos")
    args = p.parse_args()

    fijar_semilla(args.semilla)
    dispositivo = elegir_dispositivo()
    clases = leer_clases()
    print(f"Dispositivo: {dispositivo} | modelo: {args.modelo} | clases: {len(clases)}")

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    cache = os.path.join(CARPETA_RESULTADOS, f"rasgos_{args.modelo}.npz")
    splits = {nombre: leer_split(nombre) for nombre in ("train", "val", "test")}

    if os.path.exists(cache):
        print(f"Usando rasgos guardados en {cache}")
        datos = dict(np.load(cache))
    else:
        modelo = crear_modelo(args.modelo, num_clases=0, preentrenado=not args.sin_preentrenar)
        modelo.eval().to(dispositivo)
        transformacion = transformaciones(modelo, entrenamiento=False)
        datos = {}
        inicio = time.time()
        for nombre, muestras in splits.items():
            print(f"Extrayendo rasgos de {nombre} ({len(muestras)} imagenes)...")
            datos[f"X_{nombre}"], datos[f"y_{nombre}"] = extraer_rasgos(
                modelo, muestras, transformacion, dispositivo, args.lote, args.num_workers)
        np.savez(cache, **datos)
        print(f"Rasgos guardados en {cache} ({time.time() - inicio:.0f} s)")

    escala = StandardScaler().fit(datos["X_train"])
    X_train, X_val, X_test = (escala.transform(datos[f"X_{n}"]) for n in ("train", "val", "test"))
    y_train, y_val, y_test = (datos[f"y_{n}"] for n in ("train", "val", "test"))

    mejor_c, mejor_acc = None, -1.0
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=2000, random_state=args.semilla)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  C={c:<6} exactitud en validacion: {acc:.4f}")
        if acc > mejor_acc:
            mejor_c, mejor_acc = c, acc

    clf = LogisticRegression(C=mejor_c, max_iter=2000, random_state=args.semilla).fit(X_train, y_train)
    predicciones = clf.predict(X_test)
    resumen = {
        "modo": "linear_probing",
        "modelo": args.modelo,
        "C": mejor_c,
        "exactitud_val": mejor_acc,
        "exactitud_test": accuracy_score(y_test, predicciones),
        "f1_macro_test": f1_score(y_test, predicciones, average="macro"),
    }
    print(json.dumps(resumen, indent=2))

    base = os.path.join(CARPETA_RESULTADOS, f"lp_{args.modelo}_semilla{args.semilla}")
    guardar_predicciones(base + "_predicciones.csv", splits["test"], predicciones, clases)
    with open(base + "_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)
    print(f"Predicciones guardadas en {base}_predicciones.csv")


if __name__ == "__main__":
    main()
