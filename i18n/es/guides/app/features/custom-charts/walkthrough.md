---
description: Tutorial of using the custom charts feature in the W&B UI
displayed_sidebar: default
---

# Guía de Gráficos Personalizados

Para ir más allá de los gráficos integrados en W&B, utiliza la nueva característica de **Gráficos Personalizados** para controlar los detalles de exactamente qué datos estás cargando en un panel y cómo visualizar esos datos.

**Resumen**

1. Registrar datos en W&B
2. Crear una consulta
3. Personalizar el gráfico

## 1. Registrar datos en W&B

Primero, registra datos en tu script. Usa [wandb.config](../../../../guides/track/config.md) para puntos únicos establecidos al principio del entrenamiento, como hiperparámetros. Usa [wandb.log()](../../../../guides/track/log/intro.md) para múltiples puntos a lo largo del tiempo, y registra arrays 2D personalizados con wandb.Table(). Recomendamos registrar hasta 10,000 puntos de datos por clave registrada.

```python
# Registrando una tabla personalizada de datos
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

[Prueba un notebook de ejemplo rápido](https://bit.ly/custom-charts-colab) para registrar las tablas de datos, y en el siguiente paso configuraremos gráficos personalizados. Mira cómo se ven los gráficos resultantes en el [reporte en vivo](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc).

## 2. Crear una consulta

Una vez que hayas registrado datos para visualizar, ve a la página de tu proyecto y haz clic en el botón **`+`** para agregar un nuevo panel, luego selecciona **Gráfico Personalizado**. Puedes seguir el proceso en [este espacio de trabajo](https://app.wandb.ai/demo-team/custom-charts).

![Un nuevo gráfico personalizado en blanco listo para ser configurado](/images/app_ui/create_a_query.png)

### Agregar una consulta

1. Haz clic en `summary` y selecciona `historyTable` para configurar una nueva consulta que extraiga datos del historial de runs.
2. Escribe la clave donde registraste el **wandb.Table()**. En el fragmento de código anterior, fue `my_custom_table`. En el [notebook de ejemplo](https://bit.ly/custom-charts-colab), las claves son `pr_curve` y `roc_curve`.

### Configurar campos Vega

Ahora que la consulta está cargando estas columnas, están disponibles como opciones para seleccionar en los menús desplegables de campos Vega:

![Incorporando columnas de los resultados de la consulta para configurar campos Vega](/images/app_ui/set_vega_fields.png)

* **eje-x:** runSets\_historyTable\_r (recall)
* **eje-y:** runSets\_historyTable\_p (precision)
* **color:** runSets\_historyTable\_c (etiqueta de clase)

## 3. Personalizar el gráfico

Eso se ve bastante bien, pero me gustaría cambiar de un gráfico de dispersión a un gráfico de líneas. Haz clic en **Editar** para cambiar la especificación Vega de este gráfico integrado. Sigue el proceso en [este espacio de trabajo](https://app.wandb.ai/demo-team/custom-charts).

He actualizado la especificación Vega para personalizar la visualización:

* añadir títulos para el gráfico, la leyenda, el eje-x y el eje-y (establecer “title” para cada campo)
* cambiar el valor de “mark” de “point” a “line”
* eliminar el campo “size” no utilizado

Para guardar esto como un preajuste que puedes usar en otro lugar de este proyecto, haz clic en **Guardar como** en la parte superior de la página. Aquí está cómo se ve el resultado, junto con una curva ROC:

## Bonus: Histogramas Compuestos

Los histogramas pueden visualizar distribuciones numéricas para ayudarnos a entender datasets más grandes. Los histogramas compuestos muestran múltiples distribuciones a través de los mismos bins, permitiéndonos comparar dos o más métricas a través de diferentes modelos o a través de diferentes clases dentro de nuestro modelo. Para un modelo de segmentación semántica que detecta objetos en escenas de conducción, podríamos comparar la efectividad de optimizar para precisión versus intersección sobre unión (IOU), o podríamos querer saber qué tan bien diferentes modelos detectan autos (regiones grandes y comunes en los datos) versus señales de tráfico (regiones mucho más pequeñas y menos comunes). En el [Colab de demo](https://bit.ly/custom-charts-colab), puedes comparar las puntuaciones de confianza para dos de las diez clases de seres vivos.

Para crear tu propia versión del panel de histograma compuesto personalizado:

1. Crea un nuevo panel de Gráfico Personalizado en tu Espacio de Trabajo o Reporte (agregando una visualización de “Gráfico Personalizado”). Pulsa el botón “Editar” en la parte superior derecha para modificar la especificación Vega partiendo de cualquier tipo de panel integrado.
2. Reemplaza esa especificación Vega integrada con mi [código MVP para un histograma compuesto en Vega](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21). Puedes modificar el título principal, los títulos de los ejes, el dominio de entrada y cualquier otro detalle directamente en esta especificación Vega [usando la sintaxis de Vega](https://vega.github.io/) (podrías cambiar los colores o incluso agregar un tercer histograma :)
3. Modifica la consulta en el lado derecho para cargar los datos correctos de tus registros de wandb. Agrega el campo “summaryTable” y establece la “tableKey” correspondiente a “class\_scores” para obtener la wandb.Table registrada por tu run. Esto te permitirá poblar los dos conjuntos de bins del histograma (“red\_bins” y “blue\_bins”) a través de los menús desplegables con las columnas de la wandb.Table registrada como “class\_scores”. Para mi ejemplo, elegí las puntuaciones de predicción de la clase “animal” para los bins rojos y “planta” para los bins azules.
4. Puedes seguir haciendo cambios en la especificación Vega y en la consulta hasta que estés contento con el gráfico que ves en la vista previa de renderizado. Una vez que hayas terminado, haz clic en “Guardar como” en la parte superior y dale un nombre a tu gráfico personalizado para que puedas reutilizarlo. Luego haz clic en “Aplicar desde la biblioteca de paneles” para finalizar tu gráfico.

Aquí están los resultados de un experimento muy breve: entrenar con solo 1000 ejemplos durante un epoch produce un modelo que está muy confiado de que la mayoría de las imágenes no son plantas y muy incierto sobre qué imágenes podrían ser animales.