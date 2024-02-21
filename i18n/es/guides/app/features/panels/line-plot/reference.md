---
displayed_sidebar: default
---

# Referencia

## Eje X

![Seleccionando Eje X](/images/app_ui/reference_x_axis.png)

Puedes establecer el Eje X de un gráfico de líneas a cualquier valor que hayas registrado con wandb.log siempre y cuando siempre se registre como un número.

## Variables del Eje Y

Puedes establecer las variables del eje Y a cualquier valor que hayas registrado con wandb.log siempre y cuando hayas registrado números, arreglos de números o un histograma de números. Si registraste más de 1500 puntos para una variable, wandb reduce la muestra a 1500 puntos.

:::info
Puedes cambiar el color de las líneas de tu eje Y cambiando el color del run en la tabla de runs.
:::

## Rango de X y Rango de Y

Puedes cambiar los valores máximos y mínimos de X y Y para el gráfico.

El rango predeterminado de X es desde el valor más pequeño de tu eje X hasta el más grande.

El rango predeterminado de Y es desde el valor más pequeño de tus métricas y cero hasta el valor más grande de tus métricas.

## Máximo de Runs/Grupos

Por defecto, solo trazarás 10 runs o grupos de runs. Los runs se tomarán de la parte superior de tu tabla de runs o conjunto de runs, por lo que si ordenas tu tabla de runs o conjunto de runs puedes cambiar los runs que se muestran.

## Leyenda

Puedes controlar la leyenda de tu gráfico para mostrar de cualquier run cualquier valor de configuración que hayas registrado y metadatos de los runs como la hora de creación o el usuario que creó el run.

Ejemplo:

${run:displayName} - ${config:dropout} hará que el nombre de la leyenda para cada run sea algo como "royal-sweep - 0.5" donde "royal-sweep" es el nombre del run y 0.5 es el parámetro de configuración llamado "dropout".

Puedes establecer valor dentro de `[[ ]]` para mostrar valores específicos de punto en la mira al pasar el ratón sobre un gráfico. Por ejemplo, `\[\[ $x: $y ($original) ]]` mostraría algo como "2: 3 (2.9)"

Los valores admitidos dentro de \[\[ ]] son los siguientes:

| Valor       | Significado                                  |
| ----------- | -------------------------------------------- |
| ${x}        | Valor X                                      |
| ${y}        | Valor Y (Incluyendo ajuste de suavizado)     |
| ${original} | Valor Y sin incluir ajuste de suavizado      |
| ${mean}     | Media de runs agrupados                      |
| ${stddev}   | Desviación estándar de runs agrupados        |
| ${min}      | Mínimo de runs agrupados                     |
| ${max}      | Máximo de runs agrupados                     |
| ${percent}  | Porcentaje del total (para gráficos de área apilada) |

## Agrupación

Puedes agregar todos los runs activando la agrupación, o agrupar sobre una variable individual. También puedes activar la agrupación agrupando dentro de la tabla y los grupos se poblarán automáticamente en el gráfico.

## Suavizado

Puedes establecer el [coeficiente de suavizado](../../../../technical-faq/general.md#what-formula-do-you-use-for-your-smoothing-algorithm) para que esté entre 0 y 1 donde 0 es sin suavizado y 1 es el suavizado máximo.

## Ignorar Valores Atípicos

Ignorar valores atípicos hace que el gráfico establezca el mínimo y máximo del eje Y en el percentil 5 y 95 de los datos en lugar de establecerlo para hacer visible todos los datos.

## Expresión

La expresión te permite trazar valores derivados de métricas como 1-precisión. Actualmente solo funciona si estás trazando una sola métrica. Puedes hacer expresiones aritméticas simples, +, -, *, / y % así como ** para potencias.

## Estilo de Gráfico

Selecciona un estilo para tu gráfico de líneas.

**Gráfico de líneas:**

![](/images/app_ui/plot_style_line_plot.png)

**Gráfico de área:**

![](/images/app_ui/plot_style_area_plot.png)

**Gráfico de área porcentual:**

![](/images/app_ui/plot_style_percentage_plot.png)