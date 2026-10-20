# Folio

Convierte documentos escaneados en campos estructurados y buscables: OCR, extracción de campos con validación, búsqueda sobre los documentos y una API servida desde AWS Lambda detrás de API Gateway. Se construye por etapas, midiendo cada una antes de pasar a la siguiente.

[English version](README.md)

## Estado

| Etapa | Qué agrega | Estado |
|---|---|---|
| 1. Línea base de OCR | Tesseract, preprocesamiento con OpenCV, CER, WER y verificación de campos sobre recibos de SROIE | Terminada |
| 2. Línea base de extracción | Reglas para el total y la fecha, montos y fechas normalizados, verificación con el efectivo | Terminada |
| 3. Extracción aprendida y con LLM | Un clasificador de renglones, y un LLM con esquema JSON y evidencia textual | Siguiente |
| 4. Clasificación y agrupamiento | Tipo de documento y agrupamiento de formatos sin etiquetas | Planeada |
| 5. Búsqueda | BM25 y embeddings fusionados con RRF, medidos con Recall@k y MRR | Planeada |
| 6. API y AWS | FastAPI en una imagen de contenedor para Lambda detrás de API Gateway | Planeada |

<!-- results:start -->
## Resultados hasta ahora

### Etapa 2: extraer el total y la fecha

Las reglas se escribieron viendo solo 100 recibos de desarrollo y después se midieron una sola vez sobre 200 recibos reservados. Las respuestas se comparan como valores, así que `RM 5.90` y `5,90` cuentan como el mismo total. Reporte completo, con cada fallo: [`results/extract_eval.md`](results/extract_eval.md).

| Recibos | Texto que leen las reglas | Total correcto [IC 95%] | Fecha correcta [IC 95%] |
|---|---|---|---|
| 200 reservados | OCR (la mejor configuración de la etapa 1) | 151 de 200, 76% [69%, 81%] | 161 de 200, 80% [74%, 85%] |
| 200 reservados | la transcripción del propio dataset | 190 de 200, 95% [91%, 97%] | 198 de 200, 99% [96%, 100%] |
| 100 de desarrollo | OCR | 86 de 99, 87% [79%, 92%] | 84 de 100, 84% [76%, 90%] |
| 200 reservados | OCR, línea base (el monto más grande, la primera fecha) | 77 de 200, 38% [32%, 45%] | 161 de 200, 80% [74%, 85%] |

* **El cuello de botella ya es la lectura, no las reglas.** Sobre la transcripción del dataset las mismas reglas encuentran el 95% de los totales y el 99% de las fechas. Sobre el texto del OCR encuentran el 76% y el 80%. De los 49 totales que fallan con OCR, 39 son recibos que las reglas resuelven bien con el texto limpio.
* **Los recibos reservados costaron 11 puntos.** Las reglas acertaron el 87% en los recibos con los que se escribieron y el 76% en recibos nuevos. El caso más claro es el respaldo que toma el valor del renglón de abajo cuando una etiqueta de total viene sin monto: acertó 3 de 3 en desarrollo y 0 de 4 en los reservados, donde casi siempre tomó renglones de redondeo o de impuesto. Por eso el conjunto de prueba se midió una sola vez, y por eso quedan 326 recibos sin usar.
* **Efectivo menos cambio sirve como señal de confianza.** En 71 recibos reservados el total extraído coincide con el efectivo entregado menos el cambio, y es correcto en 69 (97%). En los otros 129 acierta el 64% de las veces. La etapa 3 puede mandar solo esos a un extractor más fuerte o a una persona.
* **Las etiquetas no se ponen de acuerdo con el redondeo.** La mayoría de las etiquetas de SROIE registra el monto redondeado que se pagó, pero algunas registran el total antes de redondear (12.37 en un recibo que redondea a 12.35). Ninguna regla satisface las dos convenciones, y eso pone un techo al resultado con texto limpio.
* **Para la fecha, la primera fecha válida rinde igual que cualquier regla.** Preferir la fecha junto a la etiqueta "Date" no cambió ninguna respuesta. De las 39 fechas que fallan con OCR, 37 son dígitos mal leídos o renglones ilegibles. Las otras dos fallan también con texto limpio: una fecha pegada a la palabra siguiente (`21/05/2018TIME`) y otra impresa sin separadores.

### Etapa 1: línea base de OCR

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
<!-- results:end -->

## Cómo funciona la etapa de extracción

```
palabras del OCR -> renglones -> montos y fechas como valores -> reglas de total y de fecha -> valor + renglón de origen + regla + verificación
```

* **Normalización** (`folio/normalize.py`): los montos se vuelven `Decimal` y las fechas `datetime.date`. Acepta lo que apareció en los recibos de desarrollo: la moneda pegada al número (`RM9.75`, `<RM_34.80`), coma como separador decimal, un espacio de más junto al separador (`85. 54`), un cero final perdido (`174.9`), un cero inicial perdido (`.02`) y la letra O leída en lugar de un cero. Las fechas se leen con el día primero, como se usa en Malasia, y se descartan los años inverosímiles.
* **Total** (`folio/extract.py`): un renglón de total tiene la palabra total, o una lectura del OCR a una letra de distancia, o palabras como payable, nett, grand o "after rounding". Se saltan los renglones que solo se parecen: subtotales, conteos de artículos, montos antes o sin impuesto, el impuesto mismo, ahorros y la tabla de resumen de impuestos. La lectura se detiene en el primer renglón de pago y gana el último renglón de total, porque los recibos imprimen primero el total antes de redondear. Si ese total no es múltiplo de 5 sen y le sigue un renglón de redondeo, se usa el monto redondeado. Respaldos, en orden: el valor del renglón de abajo de una etiqueta de total, efectivo menos cambio, y un pago con tarjeta.
* **La verificación con el efectivo**: cuando el recibo imprime el efectivo entregado y el cambio, su diferencia es una segunda lectura del total. Cada respuesta dice si coincide, y cuando no coinciden y la diferencia aparece impresa en el bloque de totales, gana el valor leído dos veces.
* **Fecha**: la primera fecha válida después de una etiqueta de fecha, o si no hay, la primera fecha válida, sin contar fechas de vencimiento.
* **Evaluación** (`folio/evaluate_extract.py`): cada recibo se mide sobre su texto de OCR y sobre la transcripción del dataset, contra una línea base ingenua, con intervalos de Wilson y pruebas pareadas de McNemar. Cada fallo con OCR se clasifica según si las reglas aciertan con el texto limpio, y el reporte lista todos los fallos con el renglón que usaron las reglas.
* **Caché del OCR** (`folio/cache.py`): las palabras, cajas y confianzas de cada recibo se guardan una vez en JSON, así repetir la extracción toma segundos y no minutos.

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
pytest -q                                   # 116 pruebas en unos segundos

bash scripts/get_sroie.sh                   # 626 recibos en data/sroie (unos 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # cada palabra con su caja y su confianza
folio eval-ocr --limit 100 --workers 2      # escribe results/ocr_eval.md y results/ocr_eval.csv

folio extract data/sroie/data/img/097.jpg   # total y fecha, con el renglón de donde salió cada uno
folio extract data/sroie/data/img/097.jpg --json
folio eval-extract --workers 2              # guarda el OCR en cache/ocr y escribe results/extract_eval.md y .csv
```

## Datos

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) tiene recibos escaneados de Malasia con cajas de texto por renglón y cuatro campos clave: empresa, fecha, dirección y total. El script de descarga usa el espejo [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE). Los datos son para uso de investigación y no se guardan en este repositorio. Los 626 recibos se dividen una sola vez: 100 para desarrollo (la muestra de la etapa 1), 200 reservados para prueba y 326 sin usar, guardados para una prueba nueva después de cualquier cambio a las reglas. Sus transcripciones tienen algo de ruido propio, así que ni un OCR perfecto llegaría a error cero.

## Licencia

Código bajo la [licencia MIT](LICENSE).
