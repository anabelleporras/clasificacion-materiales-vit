# Clasificación de materiales con ViT: linear probing vs. ajuste fino completo

Código del experimento de CI-0129. Compara un Vision Transformer preentrenado en dos modos:

- **Modelo base (linear probing):** el ViT queda congelado y solo se entrena una capa lineal encima.
- **Modelo ajustado (ajuste fino completo):** se entrenan todas las capas del ViT.

Funciona en Mac con chip M (usa la GPU con `mps`), en Linux con GPU NVIDIA (`cuda`) o en CPU. No hay que cambiar nada al pasar de una máquina a otra.

## Archivos

| Archivo | Qué hace |
| --- | --- |
| `preparar_datos.py` | Elige los materiales de MINC-2500 y crea las listas de entrenamiento, validación y prueba en `datos/splits/` |
| `linear_probe.py` | Modelo base: extrae los rasgos del ViT congelado una sola vez y entrena una regresión logística encima |
| `finetune.py` | Modelo ajustado: entrena todo el ViT con AdamW, parada temprana y checkpoints |
| `evaluar.py` | Exactitud, F1 macro, matrices de confusión y prueba de McNemar |
| `comun.py` | Funciones compartidas (dispositivo, datos, modelo) |

Todo lo que se genera queda en `resultados/`.

## 1. Instalar (una sola vez)

En la Terminal, dentro de esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('GPU de Apple disponible:', torch.backends.mps.is_available())"
```

Debe decir `True`. Usen Python 3.10 o más reciente.

## 2. Descargar MINC-2500

Descarguen `minc-2500.tar.gz` desde la página del dataset (http://minc.cs.cornell.edu/), descompriman el archivo y anoten la ruta. Adentro deben estar las carpetas `images/` y `labels/`.

## 3. Prueba rápida (≈ 5 % de los datos)

Sirve para verificar que todo funciona y **medir cuánto tarda una época** en su Mac antes de la corrida completa.

```bash
python preparar_datos.py --minc ~/Descargas/minc-2500 --fraccion 0.05
python linear_probe.py
python finetune.py --epocas 1 --semilla 0
```

La primera vez se descargan los pesos preentrenados del modelo (unos 90 MB). Multipliquen el tiempo de esa época por 20 para estimar una época con todos los datos.

> `linear_probe.py` guarda los rasgos en `resultados/rasgos_<modelo>.npz`. Si cambian los datos, borren ese archivo para que se vuelvan a calcular.

## 4. Corrida completa

```bash
rm -rf resultados                                   # borra los resultados de la prueba
python preparar_datos.py --minc ~/Descargas/minc-2500
python linear_probe.py
caffeinate -i python finetune.py --semilla 0         # caffeinate evita que el Mac se duerma
caffeinate -i python finetune.py --semilla 1
caffeinate -i python finetune.py --semilla 2
```

Si se interrumpe un entrenamiento, repitan el mismo comando agregando `--reanudar`.

## 5. Evaluar

```bash
python evaluar.py --base resultados/lp_*_predicciones.csv \
                  --ajustado resultados/ft_*_predicciones.csv
```

Muestra la exactitud y el F1 macro de cada corrida, el promedio ± desviación estándar entre semillas, un reporte por material, la prueba de McNemar y guarda `confusion_base.png` y `confusion_ajustado.png`.

## Opciones útiles

| Situación | Qué hacer |
| --- | --- |
| Error de memoria (`MPS backend out of memory`) | Bajen el lote: `--lote 8` |
| Mac con 16 GB y quieren un modelo más grande | `--modelo vit_base_patch16_224` en **ambos** scripts |
| Una operación no está implementada en MPS | Corran con `PYTORCH_ENABLE_MPS_FALLBACK=1 python finetune.py ...` |
| Usar otros materiales | `python preparar_datos.py --minc ... --clases wood stone fabric ...` |
| Cambiar épocas o tasa de aprendizaje | `--epocas 15 --lr 5e-5` |

Por defecto se usan 10 materiales: carpet, ceramic, fabric, glass, leather, metal, plastic, polishedstone, stone y wood. El modelo por defecto es `vit_small_patch16_224` (~22 M de parámetros), que cabe bien en un Mac de 8 GB.

**Importante:** para que la comparación sea justa, el modelo base y el ajustado deben usar el mismo `--modelo` y los mismos `datos/splits/`.
