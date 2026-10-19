# Folio

Convierte documentos escaneados en campos estructurados y buscables: OCR, extracción de campos con validación, búsqueda sobre los documentos y una API servida desde AWS Lambda detrás de API Gateway. Se construye por etapas, midiendo cada una antes de pasar a la siguiente.

[English version](README.md)

## Estado

| Etapa | Qué agrega | Estado |
|---|---|---|
| 1. Línea base de OCR | Tesseract, preprocesamiento con OpenCV, CER, WER y verificación de campos sobre recibos de SROIE | Terminada |
| 2. Línea base de extracción | Reglas con regex para fecha y total, medidas por campo | Siguiente |
| 3. Extracción aprendida y con LLM | Un clasificador de renglones, y un LLM con esquema JSON y evidencia textual | Planeada |
| 4. Clasificación y agrupamiento | Tipo de documento y agrupamiento de formatos sin etiquetas | Planeada |
| 5. Búsqueda | BM25 y embeddings fusionados con RRF, medidos con Recall@k y MRR | Planeada |
| 6. API y AWS | FastAPI en una imagen de contenedor para Lambda detrás de API Gateway | Planeada |

<!-- results:start -->
## Resultados hasta ahora: línea base de OCR

100 recibos de SROIE, los mismos para todas las configuraciones. Tabla completa: [`results/ocr_eval.md`](results/ocr_eval.md).

| Preprocesamiento | Modo de página (PSM) | CER medio [IC 95%] | F1 de palabras | Total encontrado | Segundos por recibo |
|---|---|---|---|---|---|
| grises | 6, un solo bloque | 13.2% [11.3%, 15.6%] | 0.724 | 90 de 99 | 0.80 |
| grises | 3, el de Tesseract por defecto | 17.3% [14.1%, 21.0%] | 0.706 | 79 de 99 | 0.85 |
| Otsu | 6 | 15.7% [13.5%, 18.4%] | 0.664 | 79 de 99 | 0.71 |
| adaptativa | 6 | 17.3% [15.2%, 19.6%] | 0.652 | 82 de 99 | 0.78 |
| ampliada 2x | 6 | 16.3% [13.4%, 19.9%] | 0.696 | 80 de 99 | 1.86 |

* **El modo de página pesó más que cualquier filtro de imagen.** Leer cada recibo como un solo bloque (PSM 6) bajó el CER medio 4.1 puntos contra el modo por defecto (IC 95% de 1.9 a 6.5), y encontró el total en 12 recibos donde el modo por defecto fallaba, contra 1 en sentido contrario (McNemar exacta, p = 0.003).
* **Binarizar antes de Tesseract empeoró.** Otsu subió el CER en 68 de los 100 recibos. Una explicación probable: Tesseract ya umbraliza por dentro, y un segundo umbral global puede borrar trazos tenues de la impresión térmica.
* **Ampliar al doble empeoró y tardó más del doble.** Enderezar casi no cambió nada, lo que sugiere que estos recibos se escanearon derechos.
* **Lo que todavía falla.** De los 9 totales que no encontró la mejor configuración, 4 parecen recuperables normalizando números (un espacio dentro de la cifra, un cero final perdido, un prefijo `RM_`), 2 son confusiones de caracteres (S por 5, O por 0, % por 7) y en 3 el total no se leyó de forma utilizable. Otro recibo no tiene total en su etiqueta y queda fuera. Eso define la etapa 2: un extractor que normalice los montos antes de compararlos.
<!-- results:end -->

## Cómo funciona la etapa de OCR

* **Preprocesamiento** (`folio/preprocess.py`): escala de grises, filtro de mediana, binarización de Otsu o adaptativa, ampliación al doble y enderezado por perfil de proyección. Cada paso es una función pura, y `PIPELINES` nombra las combinaciones que compara el experimento.
* **OCR** (`folio/ocr.py`): el motor LSTM de Tesseract con `image_to_data`, que conserva la caja y la confianza de cada palabra para los pasos siguientes.
* **Orden de lectura** (`folio/layout.py`): SROIE anota renglones y Tesseract devuelve palabras, a veces en otro orden. Los dos pasan por la misma regla (renglones de arriba hacia abajo, cada uno de izquierda a derecha), así el error mide la lectura y no diferencias de orden.
* **Métricas** (`folio/metrics.py`): tasas de error por carácter y por palabra a partir de la distancia de edición, F1 de bolsa de palabras, y si el total y la fecha del recibo aparecen tal cual. Se compara sin distinguir mayúsculas, porque las transcripciones de SROIE están casi todas en mayúsculas.
* **Estadística** (`folio/evaluate.py`): todas las configuraciones corren sobre los mismos recibos. Las medias llevan intervalos bootstrap de 95%, las proporciones intervalos de Wilson, y la mejor configuración se compara recibo por recibo contra la configuración por defecto de Tesseract, con una prueba exacta de McNemar sobre encontrar el total.

## Cómo correrlo

En Ubuntu o cualquier Linux con Python 3.11 o más reciente:

```bash
sudo apt install tesseract-ocr tesseract-ocr-spa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
pytest -q

bash scripts/get_sroie.sh                   # 626 recibos en data/sroie (unos 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # cada palabra con su caja y su confianza
folio eval-ocr --limit 100 --workers 2      # escribe results/ocr_eval.md y results/ocr_eval.csv
```

## Datos

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) tiene recibos escaneados de Malasia con cajas de texto por renglón y cuatro campos clave: empresa, fecha, dirección y total. El script de descarga usa el espejo [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE). Los datos son para uso de investigación y no se guardan en este repositorio. Sus transcripciones tienen algo de ruido propio, así que ni un OCR perfecto llegaría a error cero.

## Licencia

Código bajo la [licencia MIT](LICENSE).
