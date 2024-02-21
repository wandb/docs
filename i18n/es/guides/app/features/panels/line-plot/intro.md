---
description: Visualize metrics, customize axes, and compare multiple lines on the
  same plot
slug: /guides/app/features/panels/line-plot
displayed_sidebar: default
---

# Gráfico de Líneas

Los gráficos de líneas aparecen por defecto cuando graficas métricas a lo largo del tiempo con **wandb.log()**. Personaliza con configuraciones de gráficos para comparar múltiples líneas en el mismo gráfico, calcular ejes personalizados y renombrar etiquetas.

![](/images/app_ui/line_plot_example.png)

## Configuraciones

**Datos**

* **Eje X**: Selecciona ejes x por defecto incluyendo Paso y Tiempo Relativo, o selecciona un eje x personalizado. Si te gustaría usar un eje x personalizado, asegúrate de que esté registrado en la misma llamada a `wandb.log()` que usas para registrar el eje y.
  * **Tiempo Relativo (Muro)** es el tiempo de reloj desde que el proceso comenzó, así que si iniciaste un run y lo reanudaste un día después y registraste algo, eso se trazaría a las 24hrs.
  * **Tiempo Relativo (Proceso)** es el tiempo dentro del proceso en ejecución, así que si iniciaste un run y corrió por 10 segundos y lo reanudaste un día después, ese punto se trazaría a los 10s
  * **Tiempo de Muro** son los minutos transcurridos desde el inicio del primer run en el gráfico
  * **Paso** se incrementa por defecto cada vez que se llama a `wandb.log()`, y se supone que refleja el número de pasos de entrenamiento que has registrado de tu modelo
* **Ejes Y**: Selecciona ejes y de los valores registrados, incluyendo métricas e hiperparámetros que cambian a lo largo del tiempo.
* **Mínimo, máximo y escala logarítmica**: Configuraciones de mínimo, máximo y escala logarítmica para el eje x y el eje y en gráficos de líneas
* **Suavizado y excluir valores atípicos**: Cambia el suavizado en el gráfico de líneas o reescala para excluir valores atípicos de la escala mínima y máxima por defecto
* **Máximo de runs para mostrar**: Muestra más líneas en el gráfico de líneas a la vez aumentando este número, que por defecto es 10 runs. Verás el mensaje "Mostrando los primeros 10 runs" en la parte superior del gráfico si hay más de 10 runs disponibles pero el gráfico está limitando el número visible.
* **Tipo de gráfico**: Cambia entre un gráfico de líneas, un gráfico de área y un gráfico de área porcentual

**Configuraciones del Eje X**
El eje x se puede configurar a nivel del gráfico, así como globalmente para la página del proyecto o la página del reporte. Así es como se ven las configuraciones globales:

![](/images/app_ui/x_axis_global_settings.png)

:::info
Selecciona **múltiples ejes y** en las configuraciones del gráfico de líneas para comparar diferentes métricas en el mismo gráfico, como precisión y precisión de validación, por ejemplo.
:::

**Agrupamiento**

* Activa el agrupamiento para ver configuraciones para visualizar valores promediados.
* **Clave de grupo**: Selecciona una columna, y todos los runs con el mismo valor en esa columna serán agrupados juntos.
* **Agregación**: Agregación— el valor de la línea en el gráfico. Las opciones son media, mediana, min y max del grupo.
* **Rango**: Cambia el comportamiento para el área sombreada detrás de la curva agrupada. Ninguno significa que no hay área sombreada. Min/Máx muestra una región sombreada que cubre todo el rango de puntos en el grupo. Desv Est muestra la desviación estándar de los valores en el grupo. Err Est muestra el error estándar como el área sombreada.
* **Runs muestreados**: Si tienes cientos de runs seleccionados, por defecto solo muestreamos los primeros 100. Puedes seleccionar tener todos tus runs incluidos en el cálculo de agrupamiento, pero podría ralentizar las cosas en la UI.

**Leyenda**

* **Título**: Agrega un título personalizado para el gráfico de líneas, que aparece en la parte superior del gráfico
* **Título del Eje X**: Agrega un título personalizado para el eje x del gráfico de líneas, que aparece en la esquina inferior derecha del gráfico.
* **Título del Eje Y**: Agrega un título personalizado para el eje y del gráfico de líneas, que aparece en la esquina superior izquierda del gráfico.
* **Leyenda**: Selecciona el campo que quieres ver en la leyenda del gráfico para cada línea. Podrías, por ejemplo, mostrar el nombre del run y la tasa de aprendizaje.
* **Plantilla de leyenda**: Totalmente personalizable, esta poderosa plantilla te permite especificar exactamente qué texto y variables quieres mostrar en la plantilla en la parte superior del gráfico de líneas así como la leyenda que aparece cuando pasas el ratón sobre el gráfico.

![Editando la leyenda del gráfico de líneas para mostrar hiperparámetros](/images/app_ui/legend.png)

**Expresiones**

* **Expresiones del Eje Y**: Agrega métricas calculadas a tu gráfico. Puedes usar cualquiera de las métricas registradas así como valores de configuración como hiperparámetros para calcular líneas personalizadas.
* **Expresiones del Eje X**: Reescala el eje x para usar valores calculados usando expresiones personalizadas. Las variables útiles incluyen **_step** para el eje x por defecto, y la sintaxis para referenciar valores resumidos es `${summary:value}`

## Visualizar valores promedio en un gráfico

Si tienes varios experimentos diferentes y te gustaría ver el promedio de sus valores en un gráfico, puedes usar la característica de Agrupamiento en la tabla. Haz clic en "Agrupar" encima de la tabla de runs y selecciona "Todos" para mostrar valores promediados en tus gráficos.

Así es como se ve el gráfico antes de promediar:

![](/images/app_ui/demo_precision_lines.png)

Aquí he agrupado las líneas para ver el valor promedio a través de los runs.

![](/images/app_ui/demo_average_precision_lines.png)

## Visualizar valor NaN en un gráfico

También puedes trazar valores `NaN` incluyendo tensores de PyTorch en un gráfico de líneas con `wandb.log`. Por ejemplo:

```python
wandb.log({"test": [..., float("nan"), ...]})
```

![](/images/app_ui/visualize_nan.png)

## Comparar dos métricas en un mismo gráfico

Haz clic en un run para ir a la página del run. Aquí hay un [ejemplo de run](https://app.wandb.ai/stacey/estuary/runs/9qha4fuu?workspace=user-carey) del proyecto Estuary de Stacey. Los gráficos generados automáticamente muestran métricas individuales.


![](@site/static/images/app_ui/visualization_add.png)

Haz clic **en el signo más** en la parte superior derecha de la página, y selecciona el **Gráfico de Líneas**.

![](https://downloads.intercomcdn.com/i/o/142936481/d0648728180887c52ab46549/image.png)

En el campo **Variables Y**, selecciona algunas métricas que te gustaría comparar. Aparecerán juntas en el gráfico de líneas.

![](https://downloads.intercomcdn.com/i/o/146033909/899fc05e30795a1d7699dc82/Screen+Shot+2019-09-04+at+9.10.52+AM.png)

## Cambiar el color de los gráficos de líneas

A veces, el color predeterminado de los runs no es útil para la comparación. Para ayudar a superar esto, wandb proporciona dos instancias con las que se puede cambiar manualmente los colores.

### Desde la tabla de runs

A cada run se le da un color aleatorio por defecto al inicializar.

![Colores aleatorios dados a los runs](/images/app_ui/line_plots_run_table_random_colors.png)

Al hacer clic en cualquiera de los colores, aparece una paleta de colores de la cual podemos elegir manualmente el color que queremos.

![La paleta de colores](/images/app_ui/line_plots_run_table_color_palette.png)

### Desde las configuraciones de leyenda del gráfico

También se puede cambiar el color de los runs desde las configuraciones de leyenda del gráfico.


![](/images/app_ui/plot_style_line_plot_legend.png)

## Visualizar en diferentes ejes x

Si te gustaría ver el tiempo absoluto que ha tomado un experimento, o ver qué día se realizó un experimento, puedes cambiar el eje x. Aquí hay un ejemplo de cambio de pasos a tiempo relativo y luego a tiempo de muro.

![](/images/app_ui/howto_use_relative_time_or_wall_time.gif)

## Gráficos de área

En las configuraciones del gráfico de líneas, en la pestaña avanzada, haz clic en diferentes estilos de gráfico para obtener un gráfico de área o un gráfico de área porcentual.

![](/images/app_ui/line_plots_area_plots.gif)

## Zoom

Haz clic y arrastra un rectángulo para hacer zoom vertical y horizontalmente al mismo tiempo. Esto cambia el zoom del eje x y el eje y.

![](/images/app_ui/line_plots_zoom.gif)

## Ocultar leyenda del gráfico

Apaga la leyenda en el gráfico de líneas con este simple interruptor:

![](/images/app_ui/demo_hide_legend.gif)