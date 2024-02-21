---
description: In line plots, use smoothing to see trends in noisy data.
displayed_sidebar: default
---

# Suavizado

En las gráficas de línea de W&B, soportamos tres tipos de suavizado:

- [media móvil exponencial](smoothing.md#exponential-moving-average-default) (por defecto)
- [suavizado gaussiano](smoothing.md#gaussian-smoothing)
- [media móvil](smoothing.md#running-average)
- [media móvil exponencial - Tensorboard](smoothing.md#exponential-moving-average-tensorboard) (obsoleto)

Vea estos en un [reporte interactivo de W&B](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc).

![](/images/app_ui/beamer_smoothing.gif)

## Media Móvil Exponencial (Por defecto)

El suavizado exponencial es una técnica para suavizar datos de series de tiempo mediante el decaimiento exponencial del peso de puntos anteriores. El rango es de 0 a 1. Vea [Suavizado Exponencial](https://www.wikiwand.com/en/Exponential_smoothing) para antecedentes. Se añade un término de des-sesgo para que los valores tempranos en las series de tiempo no estén sesgados hacia cero.

El algoritmo EMA toma en cuenta la densidad de puntos en la línea (es decir, el número de valores de `y` por unidad de rango en el eje x). Esto permite un suavizado consistente al mostrar múltiples líneas con diferentes características simultáneamente.

Aquí hay un código de ejemplo de cómo funciona esto internamente:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE escala el resultado al rango del eje x del gráfico
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

Así es como se ve [en la aplicación](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/weighted_exponential_moving_average.png)

## Suavizado Gaussiano

El suavizado gaussiano (o suavizado de núcleo gaussiano) calcula un promedio ponderado de los puntos, donde los pesos corresponden a una distribución gaussiana con la desviación estándar especificada como el parámetro de suavizado. Vea . El valor suavizado se calcula para cada valor de entrada x.

El suavizado gaussiano es una buena elección estándar para suavizar si no le preocupa coincidir con el comportamiento de TensorBoard. A diferencia de una media móvil exponencial, el punto se suavizará basándose en puntos que ocurren tanto antes como después del valor.

Así es como se ve [en la aplicación](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

![](/images/app_ui/gaussian_smoothing.png)

## Media Móvil

La media móvil es un algoritmo de suavizado que reemplaza un punto con el promedio de puntos en una ventana antes y después del valor de x dado. Vea "Filtro de Vagón" en [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average). El parámetro seleccionado para la media móvil le dice a Pesos y Sesgos el número de puntos a considerar en la media móvil.

Considere usar Suavizado Gaussiano si sus puntos están espaciados de forma desigual en el eje x.

La siguiente imagen demuestra cómo se ve una aplicación de ejecución [en la aplicación](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

![](/images/app_ui/running_average.png)

## Media Móvil Exponencial (Obsoleto)

> El algoritmo EMA de TensorBoard ha sido obsoleto ya que no puede suavizar de manera precisa múltiples líneas en el mismo gráfico que no tienen una densidad de puntos consistente (número de puntos trazados por unidad del eje x).

La media móvil exponencial se implementa para coincidir con el algoritmo de suavizado de TensorBoard. El rango es de 0 a 1. Vea [Suavizado Exponencial](https://www.wikiwand.com/en/Exponential_smoothing) para antecedentes. Se añade un término de des-sesgo para que los valores tempranos en las series de tiempo no estén sesgados hacia cero.

Aquí hay un código de ejemplo de cómo funciona esto internamente:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

Así es como se ve [en la aplicación](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/exponential_moving_average.png)

## Detalles de Implementación

Todos los algoritmos de suavizado funcionan en los datos muestreados, lo que significa que si registra más de 1500 puntos, el algoritmo de suavizado se ejecutará _después_ de que los puntos se descarguen del servidor. La intención de los algoritmos de suavizado es ayudar a encontrar patrones en los datos rápidamente. Si necesita valores suavizados exactos en métricas con un gran número de puntos registrados, puede ser mejor descargar sus métricas a través de la API y ejecutar sus propios métodos de suavizado.

## Ocultar datos originales

Por defecto mostramos los datos originales, sin suavizar, como una línea tenue en el fondo. Haga clic en el interruptor **Mostrar Original** para desactivar esto.

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)