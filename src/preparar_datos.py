"""Crea las listas de entrenamiento, validacion y prueba a partir de MINC-2500.

Estructura esperada de MINC-2500 (tal como se descarga):
    minc-2500/
        images/<material>/<material>_XXXXXX.jpg
        labels/train1.txt, validate1.txt, test1.txt, ... (particiones oficiales 1 a 5)

Si existen las particiones oficiales se usan (recomendado: permiten comparar con
otros trabajos). Si no, se hace una division estratificada 70/15/15.

Ejemplos:
    python preparar_datos.py --minc ~/Datos/minc-2500 /Users/lulu/Downloads/minc-2500
    python preparar_datos.py --minc ~/Datos/minc-2500 --fraccion 0.1   # prueba rapida
"""

import argparse
import csv
import os
import random
from collections import defaultdict

from comun import CARPETA_SPLITS

CLASES_POR_DEFECTO = ["metal", "stone", "wood"]


def leer_particion_oficial(minc, archivo, clases):
    muestras = []
    with open(os.path.join(minc, "labels", archivo), encoding="utf-8") as f:
        for linea in f:
            relativa = linea.strip()
            if not relativa:
                continue
            clase = relativa.split("/")[1]
            if clase in clases:
                muestras.append((os.path.join(minc, relativa), clase))
    return muestras


def division_estratificada(minc, clases, semilla):
    rng = random.Random(semilla)
    splits = {"train": [], "val": [], "test": []}
    for clase in clases:
        carpeta = os.path.join(minc, "images", clase)
        archivos = sorted(a for a in os.listdir(carpeta) if a.lower().endswith((".jpg", ".jpeg", ".png")))
        rng.shuffle(archivos)
        n = len(archivos)
        n_train, n_val = int(0.70 * n), int(0.15 * n)
        partes = {
            "train": archivos[:n_train],
            "val": archivos[n_train:n_train + n_val],
            "test": archivos[n_train + n_val:],
        }
        for nombre, lista in partes.items():
            splits[nombre] += [(os.path.join(carpeta, a), clase) for a in lista]
    return splits


def submuestrear(muestras, fraccion, rng):
    """Toma la misma fraccion de cada clase (para pruebas rapidas)."""
    por_clase = defaultdict(list)
    for ruta, clase in muestras:
        por_clase[clase].append((ruta, clase))
    resultado = []
    for clase, lista in sorted(por_clase.items()):
        rng.shuffle(lista)
        resultado += lista[:max(1, int(len(lista) * fraccion))]
    return resultado


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minc", required=True, help="Carpeta raiz de MINC-2500")
    p.add_argument("--clases", nargs="+", default=CLASES_POR_DEFECTO, help="Materiales a usar")
    p.add_argument("--particion", type=int, default=1, help="Particion oficial 1-5 (si existe)")
    p.add_argument("--fraccion", type=float, default=1.0, help="Fraccion de imagenes por clase (ej. 0.1 para pruebas)")
    p.add_argument("--semilla", type=int, default=0)
    args = p.parse_args()

    minc = os.path.expanduser(args.minc)
    clases = sorted(args.clases)
    oficial = os.path.join(minc, "labels", f"train{args.particion}.txt")

    if os.path.exists(oficial):
        print(f"Usando la particion oficial {args.particion} de MINC-2500.")
        splits = {
            "train": leer_particion_oficial(minc, f"train{args.particion}.txt", clases),
            "val": leer_particion_oficial(minc, f"validate{args.particion}.txt", clases),
            "test": leer_particion_oficial(minc, f"test{args.particion}.txt", clases),
        }
    else:
        print("No se encontraron particiones oficiales: division estratificada 70/15/15.")
        splits = division_estratificada(minc, clases, args.semilla)

    if args.fraccion < 1.0:
        rng = random.Random(args.semilla)
        splits = {k: submuestrear(v, args.fraccion, rng) for k, v in splits.items()}

    os.makedirs(CARPETA_SPLITS, exist_ok=True)
    with open(os.path.join(CARPETA_SPLITS, "clases.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(clases) + "\n")
    for nombre, muestras in splits.items():
        with open(os.path.join(CARPETA_SPLITS, f"{nombre}.csv"), "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(["ruta", "clase"])
            escritor.writerows(muestras)
        print(f"  {nombre}: {len(muestras)} imagenes")
    print(f"Clases ({len(clases)}): {', '.join(clases)}")


if __name__ == "__main__":
    main()
