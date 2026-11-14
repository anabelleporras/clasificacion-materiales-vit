"""Modelo ajustado: ajuste fino completo del ViT.

Entrena TODAS las capas con AdamW y una tasa de aprendizaje baja. Usa el
conjunto de validacion para guardar el mejor modelo y detenerse si deja de
mejorar (parada temprana). Guarda un checkpoint en cada epoca: si se
interrumpe, vuelva a correr el mismo comando con --reanudar.

Ejemplos:
    python finetune.py --modelo vit_small_patch16_224 --semilla 0
    python finetune.py --modelo vit_small_patch16_224 --semilla 0 --reanudar
"""

import argparse
import json
import os
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comun import (CARPETA_RESULTADOS, DatasetMateriales, crear_modelo, elegir_dispositivo,
                   fijar_semilla, guardar_predicciones, leer_clases, leer_split, transformaciones)


def entrenar_epoca(modelo, cargador, criterio, optimizador, dispositivo):
    modelo.train()
    perdida_total, n = 0.0, 0
    for imagenes, etiquetas in tqdm(cargador, desc="  entrenando", leave=False):
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(imagenes), etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(etiquetas)
        n += len(etiquetas)
    return perdida_total / n


@torch.no_grad()
def predecir(modelo, cargador, dispositivo):
    modelo.eval()
    predicciones, reales = [], []
    for imagenes, etiquetas in tqdm(cargador, desc="  evaluando", leave=False):
        salida = modelo(imagenes.to(dispositivo))
        predicciones += salida.argmax(dim=1).cpu().tolist()
        reales += etiquetas.tolist()
    return reales, predicciones


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", default="vit_small_patch16_224", help="Nombre del modelo en timm")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--lote", type=int, default=16, help="Bajelo a 8 si se queda sin memoria")
    p.add_argument("--lr", type=float, default=3e-5, help="Tasa de aprendizaje (baja para no destruir lo preentrenado)")
    p.add_argument("--decaimiento", type=float, default=0.05, help="Weight decay de AdamW")
    p.add_argument("--paciencia", type=int, default=3, help="Epocas sin mejorar antes de detenerse")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--reanudar", action="store_true", help="Continua desde el ultimo checkpoint")
    p.add_argument("--sin-preentrenar", action="store_true", help="Solo para probar el codigo sin descargar pesos")
    args = p.parse_args()

    fijar_semilla(args.semilla)
    dispositivo = elegir_dispositivo()
    clases = leer_clases()
    print(f"Dispositivo: {dispositivo} | modelo: {args.modelo} | clases: {len(clases)}")

    modelo = crear_modelo(args.modelo, num_clases=len(clases), preentrenado=not args.sin_preentrenar)
    modelo.to(dispositivo)

    entrenamiento, validacion, prueba = (leer_split(n) for n in ("train", "val", "test"))
    t_train = transformaciones(modelo, entrenamiento=True)
    t_eval = transformaciones(modelo, entrenamiento=False)
    generador = torch.Generator().manual_seed(args.semilla)
    cargar = lambda muestras, t, mezclar: DataLoader(  # noqa: E731
        DatasetMateriales(muestras, t), batch_size=args.lote, shuffle=mezclar,
        num_workers=args.num_workers, generator=generador if mezclar else None,
        persistent_workers=args.num_workers > 0)
    c_train = cargar(entrenamiento, t_train, True)
    c_val = cargar(validacion, t_eval, False)
    c_test = cargar(prueba, t_eval, False)

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.decaimiento)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    base = os.path.join(CARPETA_RESULTADOS, f"ft_{args.modelo}_semilla{args.semilla}")
    ruta_ultimo, ruta_mejor = base + "_ultimo.pt", base + "_mejor.pt"

    epoca_inicial, mejor_acc, sin_mejorar, historial = 0, -1.0, 0, []
    if args.reanudar and os.path.exists(ruta_ultimo):
        estado = torch.load(ruta_ultimo, map_location=dispositivo)
        modelo.load_state_dict(estado["modelo"])
        optimizador.load_state_dict(estado["optimizador"])
        planificador.load_state_dict(estado["planificador"])
        epoca_inicial, mejor_acc = estado["epoca"] + 1, estado["mejor_acc"]
        sin_mejorar, historial = estado["sin_mejorar"], estado["historial"]
        print(f"Reanudando desde la epoca {epoca_inicial + 1}")

    for epoca in range(epoca_inicial, args.epocas):
        inicio = time.time()
        perdida = entrenar_epoca(modelo, c_train, criterio, optimizador, dispositivo)
        planificador.step()
        reales, preds = predecir(modelo, c_val, dispositivo)
        acc_val = accuracy_score(reales, preds)
        historial.append({"epoca": epoca + 1, "perdida_train": perdida, "exactitud_val": acc_val})
        print(f"Epoca {epoca + 1}/{args.epocas}  perdida={perdida:.4f}  exactitud_val={acc_val:.4f}  "
              f"({time.time() - inicio:.0f} s)")

        if acc_val > mejor_acc:
            mejor_acc, sin_mejorar = acc_val, 0
            torch.save(modelo.state_dict(), ruta_mejor)
        else:
            sin_mejorar += 1

        torch.save({"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                    "planificador": planificador.state_dict(), "epoca": epoca, "mejor_acc": mejor_acc,
                    "sin_mejorar": sin_mejorar, "historial": historial}, ruta_ultimo)

        if sin_mejorar >= args.paciencia:
            print(f"Parada temprana: {args.paciencia} epocas sin mejorar en validacion.")
            break

    modelo.load_state_dict(torch.load(ruta_mejor, map_location=dispositivo))
    reales, predicciones = predecir(modelo, c_test, dispositivo)
    resumen = {
        "modo": "ajuste_fino_completo",
        "modelo": args.modelo,
        "lr": args.lr,
        "lote": args.lote,
        "mejor_exactitud_val": mejor_acc,
        "exactitud_test": accuracy_score(reales, predicciones),
        "f1_macro_test": f1_score(reales, predicciones, average="macro"),
        "historial": historial,
    }
    print(json.dumps({k: v for k, v in resumen.items() if k != "historial"}, indent=2))

    guardar_predicciones(base + "_predicciones.csv", prueba, predicciones, clases)
    with open(base + "_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)
    print(f"Predicciones guardadas en {base}_predicciones.csv")


if __name__ == "__main__":
    main()
