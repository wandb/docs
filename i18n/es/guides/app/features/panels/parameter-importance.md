---
description: Visualize the relationships between your model's hyperparameters and
  output metrics
displayed_sidebar: default
---

# Importancia de los parámetros

Este panel muestra cuáles de tus hiperparámetros fueron los mejores predictores de, y altamente correlacionados con valores deseables de tus métricas.

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795733856\_image.png)

**Correlación** es la correlación lineal entre el hiperparámetro y la métrica elegida (en este caso val\_loss). Por lo tanto, una alta correlación significa que cuando el hiperparámetro tiene un valor más alto, la métrica también tiene valores más altos y viceversa. La correlación es una gran métrica para observar, pero no puede capturar interacciones de segundo orden entre entradas y puede ser complicado comparar entradas con rangos muy diferentes.

Por lo tanto, también calculamos una métrica de **importancia** donde entrenamos un bosque aleatorio con los hiperparámetros como entradas y la métrica como salida objetivo e informamos los valores de importancia de las características para el bosque aleatorio.

La idea para esta técnica fue inspirada por una conversación con [Jeremy Howard](https://twitter.com/jeremyphoward) quien ha sido pionero en el uso de importancias de características del bosque aleatorio para explorar espacios de hiperparámetros en [Fast.ai](http://fast.ai). Te recomendamos encarecidamente que revises su fenomenal [conferencia](http://course18.fast.ai/lessonsml1/lesson4.html) (y estas [notas](https://forums.fast.ai/t/wiki-lesson-thread-lesson-4/7540)) para aprender más sobre la motivación detrás de este análisis.

Este panel de importancia de hiperparámetros desenreda las complicadas interacciones entre hiperparámetros altamente correlacionados. Al hacerlo, te ayuda a afinar tus búsquedas de hiperparámetros mostrándote cuáles de tus hiperparámetros son los más importantes en términos de predecir el rendimiento del modelo.

## Creando un Panel de Importancia de Hiperparámetros

Ve a tu Proyecto W&B. Si no tienes uno, puedes usar [este proyecto](https://app.wandb.ai/sweep/simpsons).

Desde la página de tu proyecto, haz clic en **Agregar Visualización**.

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795570241\_image.png)

Luego elige **Importancia de Parámetros**.

No necesitas escribir ningún código nuevo, aparte de [integrar W&B](https://docs.wandb.com/quickstart) en tu proyecto.

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578795636072\_image.png)

:::info
Si aparece un panel vacío, asegúrate de que tus runs estén desagrupados
:::

## Usando el Panel de Importancia de Hiperparámetros

Podemos dejar que wandb visualice el conjunto más útil de hiperparámetros haciendo clic en la varita mágica al lado del gestor de parámetros. Luego podemos ordenar los hiperparámetros basándonos en la Importancia.

![Usando visualización automática de parámetros](/images/app_ui/hyperparameter_importance_panel.gif)

Con el gestor de parámetros, podemos configurar manualmente los parámetros visibles y ocultos.

![Configurando manualmente los campos visibles y ocultos](/images/app_ui/hyperparameter_importance_panel_manual.gif)

## Interpretando un Panel de Importancia de Hiperparámetros

![](https://paper-attachments.dropbox.com/s\_B78AACEDFC4B6CE0BF245AA5C54750B01173E5A39173E03BE6F3ACF776A01267\_1578798509642\_image.png)

Este panel te muestra todos los parámetros pasados al objeto [wandb.config](https://docs.wandb.com/library/python/config) en tu script de entrenamiento. A continuación, muestra las importancias de las características y las correlaciones de estos parámetros de configuración con respecto a la métrica del modelo que selecciones (`val_loss` en este caso).

### Importancia

La columna de importancia te muestra el grado en el que cada hiperparámetro fue útil para predecir la métrica elegida. Podemos imaginar un escenario en el que comenzamos ajustando una plétora de hiperparámetros y usando este gráfico para centrarnos en cuáles merecen una exploración más profunda. Los barridos subsiguientes entonces pueden limitarse a los hiperparámetros más importantes, encontrando así un mejor modelo más rápido y de forma más económica.

Nota: Calculamos estas importancias usando un modelo basado en árboles en lugar de un modelo lineal ya que los primeros son más tolerantes tanto con los datos categóricos como con los datos que no están normalizados.\
En el panel mencionado podemos ver que `epochs, learning_rate, batch_size` y `weight_decay` fueron bastante importantes.

Como siguiente paso, podríamos realizar otro barrido explorando valores más detallados de estos hiperparámetros. Curiosamente, mientras que `learning_rate` y `batch_size` eran importantes, no estaban muy bien correlacionados con la salida.\
Esto nos lleva a las correlaciones.

### Correlaciones

Las correlaciones capturan relaciones lineales entre hiperparámetros individuales y valores de métricas. Responden a la pregunta: ¿existe una relación significativa entre usar un hiperparámetro, digamos el optimizador SGD, y mi val\_loss (la respuesta en este caso es sí)? Los valores de correlación van de -1 a 1, donde los valores positivos representan correlación lineal positiva, los valores negativos representan correlación lineal negativa y un valor de 0 representa ninguna correlación. Generalmente, un valor mayor a 0.7 en cualquier dirección representa una fuerte correlación.

Podríamos usar este gráfico para explorar más los valores que tienen una mayor correlación con nuestra métrica (en este caso podríamos elegir descenso de gradiente estocástico o adam sobre rmsprop o nadam) o entrenar durante más epochs.

Nota rápida sobre la interpretación de correlaciones:

* las correlaciones muestran evidencia de asociación, no necesariamente de causalidad.
* las correlaciones son sensibles a los valores atípicos, lo que podría convertir una relación fuerte en una moderada, especialmente si el tamaño de la muestra de hiperparámetros probados es pequeño.
* y finalmente, las correlaciones solo capturan relaciones lineales entre hiperparámetros y métricas. Si hay una fuerte relación polinomial, no será capturada por las correlaciones.

Las disparidades entre importancia y correlaciones resultan del hecho de que la importancia tiene en cuenta las interacciones entre hiperparámetros, mientras que la correlación solo mide los efectos de hiperparámetros individuales sobre los valores de las métricas. En segundo lugar, las correlaciones capturan solo las relaciones lineales, mientras que las importancias pueden capturar otras más complejas.

Como puedes ver, tanto la importancia como las correlaciones son herramientas poderosas para entender cómo tus hiperparámetros influencian el rendimiento del modelo.

Esperamos que este panel te ayude a capturar estos conocimientos y a enfocarte en un modelo potente más rápidamente.