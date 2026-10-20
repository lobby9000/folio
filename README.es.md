# Folio

Convierte documentos escaneados en campos estructurados y buscables: OCR, extracción de campos con validación, búsqueda sobre los documentos y una API servida desde AWS Lambda detrás de API Gateway. Se construye por etapas, midiendo cada una antes de pasar a la siguiente.

[English version](README.md)

## Estado

| Etapa | Qué agrega | Estado |
|---|---|---|
| 1. Línea base de OCR | Tesseract, preprocesamiento con OpenCV, CER, WER y verificación de campos sobre recibos de SROIE | Terminada |
| 2. Línea base de extracción | Reglas para el total y la fecha, montos y fechas normalizados, verificación con el efectivo | Terminada |
| 3. Extracción aprendida y con LLM | Un ranker aprendido de recibos etiquetados, un LLM con esquema JSON y evidencia textual, y una cascada que solo llama al LLM cuando hay duda | Ranker terminado, falta correr el LLM |
| 4. Clasificación y agrupamiento | Tipo de documento y agrupamiento de formatos sin etiquetas | Siguiente |
| 5. Búsqueda | BM25 y embeddings fusionados con RRF, medidos con Recall@k y MRR | Planeada |
| 6. API y AWS | FastAPI en una imagen de contenedor para Lambda detrás de API Gateway | Planeada |

<!-- results:start -->
## Resultados hasta ahora

### Etapa 3: un ranker aprendido y la cascada con LLM

Cada monto que el OCR encuentra en un recibo se vuelve un candidato con 42 rasgos, y un modelo aprende de recibos etiquetados cuál de ellos es el total. Se ajustó con 426 recibos (los 100 de desarrollo de la etapa 2 más 326 que nadie había usado), se eligió con validación cruzada sobre ellos y se midió una sola vez sobre los mismos 200 recibos reservados que las reglas. Reporte completo: [`results/learned_eval.md`](results/learned_eval.md).

| Sobre los 200 recibos reservados, texto del OCR | Total correcto [IC 95%] |
|---|---|
| Reglas (etapa 2) | 151 de 200, 76% [69%, 81%] |
| Ranker aprendido, regresión logística | 163 de 200, 82% [76%, 86%] |
| Techo: el total está entre los montos que leyó el OCR | 176 de 200, 88% [83%, 92%] |

* **El ranker le gana a las reglas en los mismos recibos.** Acierta en 15 recibos donde las reglas fallan, contra 3 al revés (McNemar exacta, p = 0.0075). La mayoría de esos 15 son recibos donde ninguna regla dio respuesta.
* **Bastó un modelo simple.** La regresión logística y los árboles con gradient boosting empataron en validación cruzada (80.6% y 80.5% en 15 particiones), así que se quedó la logística: sus 42 pesos caben en un JSON y se pueden leer. Los más pesados dicen que el total suele ser el monto más grande, el que eligieron las reglas, un múltiplo de 5 sen (en Malasia las cuentas se redondean a 5 sen) e igual a efectivo menos cambio.
* **Su puntaje sirve como confianza.** En la mitad más segura de los recibos reservados acierta 98 de 100.
* **Eso decide cuándo llamar a un LLM.** La cascada se queda con el total de las reglas cuando coincide con efectivo menos cambio (71 recibos, 97% correctos) y con el del ranker cuando su puntaje es al menos 0.95 (53 recibos, 87% correctos), un umbral elegido con los recibos de entrenamiento. Solo los otros 76 irían al LLM, y en esos el ranker solo acierta el 66%.
* **El extractor con LLM está construido y probado, todavía no medido.** Responde con un esquema JSON que incluye el renglón de donde sale cada valor, y un valor solo se acepta si está impreso en el texto del OCR, así que el modelo puede elegir pero no inventar. Corre con la API de Claude o con un modelo local en Ollama. Sus cifras se agregan después de correrlo sobre los recibos que la cascada le manda.
* Las reglas, medidas sobre los 326 recibos que aquí se usaron para entrenar, encuentran el 73% de los totales, cerca de su 76% en prueba. Es una segunda señal de que el resultado de prueba no fue suerte.

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

## Cómo funciona la etapa aprendida

```
montos del recibo -> 42 rasgos cada uno -> regresión logística -> mejor candidato + puntaje
      ¿reglas confirmadas por el efectivo? -> se queda el total de las reglas
      ¿puntaje >= 0.95?                     -> se queda el total del ranker
      si no                                  -> LLM con esquema JSON -> su total solo si está impreso en el texto
```

* **Candidatos y rasgos** (`folio/learn.py`): cada monto de cada renglón. Los rasgos describen el renglón (palabras de total, subtotal, impuesto, efectivo y cambio, con las lecturas del OCR que las reglas ya aceptan), los renglones vecinos, su lugar en la página, la altura del renglón, la confianza del OCR, cuántas veces se repite el mismo valor, si es múltiplo de 5 sen, si es igual a efectivo menos cambio y si las reglas lo eligieron. La evidencia también se junta por valor, porque el mismo monto impreso en el renglón del total y en el del efectivo es una sola respuesta vista dos veces.
* **Elección del modelo**: regresión logística y árboles con gradient boosting, comparados con validación cruzada de 5 particiones repetida tres veces, siempre separando recibos completos para que ningún recibo ayude a calificarse a sí mismo. El umbral de la cascada se eligió con predicciones fuera de partición sobre los recibos de entrenamiento, nunca con los de prueba.
* **Extractor con LLM** (`folio/llm.py`): los renglones del OCR van numerados al modelo, que debe responder con el valor y el número de renglón del total y la fecha, mediante salidas estructuradas. La respuesta se verifica contra el texto: un valor que no está impreso en ningún lado se rechaza y se queda la respuesta más barata. Las respuestas se guardan en disco con un hash del prompt.
* **Evaluación** (`folio/evaluate_learned.py`, `folio/evaluate_llm.py`): comparaciones pareadas sobre los mismos recibos, intervalos de Wilson, exactitud por nivel de confianza, las rutas de la cascada y las respuestas del LLM por estado (con evidencia, renglón equivocado, no está en el texto).

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
pytest -q                                   # 138 pruebas en unos segundos

bash scripts/get_sroie.sh                   # 626 recibos en data/sroie (unos 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # cada palabra con su caja y su confianza
folio eval-ocr --limit 100 --workers 2      # escribe results/ocr_eval.md y results/ocr_eval.csv

folio extract data/sroie/data/img/097.jpg   # total y fecha, con el renglón de donde salió cada uno
folio extract data/sroie/data/img/097.jpg --json
folio eval-extract --workers 2              # guarda el OCR en cache/ocr y escribe results/extract_eval.md y .csv

folio eval-learned                          # validación cruzada, una corrida de prueba, guarda models/total_ranker.json
folio extract data/sroie/data/img/097.jpg --ranker models/total_ranker.json

# La etapa con LLM. Con un modelo local (nada sale de tu máquina):
ollama pull qwen2.5:3b
folio eval-llm --backend ollama --model qwen2.5:3b          # solo pregunta por los recibos que manda la cascada
# O con la API de Claude:
pip install anthropic && export ANTHROPIC_API_KEY=...
folio eval-llm --backend anthropic --model claude-haiku-5-5 --mode all   # todos los recibos, para medirlo solo
```

## Datos

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) tiene recibos escaneados de Malasia con cajas de texto por renglón y cuatro campos clave: empresa, fecha, dirección y total. El script de descarga usa el espejo [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE). Los datos son para uso de investigación y no se guardan en este repositorio. Los 626 recibos se dividen una sola vez: 100 para desarrollo (la muestra de la etapa 1), 200 reservados para prueba y los otros 326 para entrenamiento. Esos 326 no se usaron en la etapa 2 y en la etapa 3 se volvieron los datos de entrenamiento del ranker, así que los cambios futuros se juzgan con validación cruzada sobre ellos y con la única corrida de prueba que recibe cada método. Sus transcripciones tienen algo de ruido propio, así que ni un OCR perfecto llegaría a error cero.

## Licencia

Código bajo la [licencia MIT](LICENSE).
