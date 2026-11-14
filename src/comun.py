"""Funciones compartidas: dispositivo, semillas, datos, transformaciones y modelo."""

import csv
import os
import random

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CARPETA_SPLITS = os.path.join("datos", "splits")
CARPETA_RESULTADOS = "resultados"


def elegir_dispositivo():
    """GPU NVIDIA (cuda) > GPU de Apple (mps) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def leer_clases():
    with open(os.path.join(CARPETA_SPLITS, "clases.txt"), encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


def leer_split(nombre):
    """Devuelve una lista de (ruta_imagen, indice_clase) para 'train', 'val' o 'test'."""
    clases = leer_clases()
    indice = {c: i for i, c in enumerate(clases)}
    with open(os.path.join(CARPETA_SPLITS, f"{nombre}.csv"), encoding="utf-8") as f:
        return [(fila["ruta"], indice[fila["clase"]]) for fila in csv.DictReader(f)]


class DatasetMateriales(Dataset):
    def __init__(self, muestras, transformacion):
        self.muestras = muestras
        self.transformacion = transformacion

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, etiqueta = self.muestras[i]
        imagen = Image.open(ruta).convert("RGB")
        return self.transformacion(imagen), etiqueta


def crear_modelo(nombre, num_clases, preentrenado=True):
    """num_clases=0 devuelve solo el extractor de rasgos (sin cabeza)."""
    return timm.create_model(nombre, pretrained=preentrenado, num_classes=num_clases)


def transformaciones(modelo, entrenamiento):
    """Usa el tamano, la media y la desviacion con que se preentreno el modelo.

    En entrenamiento: recortes aleatorios y volteo horizontal. No se altera el
    color ni el brillo, porque son pistas del material.
    """
    config = timm.data.resolve_model_data_config(modelo)
    tamano = config["input_size"][-1]
    normalizar = transforms.Normalize(mean=config["mean"], std=config["std"])
    if entrenamiento:
        return transforms.Compose([
            transforms.RandomResizedCrop(tamano, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizar,
        ])
    return transforms.Compose([
        transforms.Resize(tamano),
        transforms.CenterCrop(tamano),
        transforms.ToTensor(),
        normalizar,
    ])


def guardar_predicciones(ruta_csv, muestras, predicciones, clases):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["ruta", "real", "prediccion"])
        for (ruta, real), pred in zip(muestras, predicciones):
            escritor.writerow([ruta, clases[real], clases[int(pred)]])
