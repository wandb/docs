---
displayed_sidebar: default
---

# Muestreo y Agrupación en Buckets

## Muestreo

Por razones de rendimiento, cuando se eligen más de 1500 puntos para una métrica de gráfico de líneas, W&B devuelve 1500 puntos muestreados aleatoriamente. Cada métrica se muestrea por separado y solo se consideran los pasos donde la métrica se registra realmente.

Si quieres ver todas las métricas registradas para un run o implementar tu propio muestreo, puedes usar la API de W&B.

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

## Agrupación en Buckets

Cuando se agrupan o se utilizan expresiones con múltiples runs con valores del eje x posiblemente no alineados, se utiliza la agrupación en buckets para reducir la cantidad de puntos. El eje x se divide en 200 segmentos de igual tamaño y luego, dentro de cada segmento, se promedian todos los puntos para una métrica dada. Cuando se agrupan o se utilizan expresiones para combinar métricas, este promedio dentro de un segmento se utiliza como el valor de la métrica.