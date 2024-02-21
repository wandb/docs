---
description: Visualize the results of your W&B Sweeps with the W&B App UI.
displayed_sidebar: default
---

# Visualiza los resultados de un barrido

<head>
  <title>Visualiza Resultados de los Barridos de W&B</title>
</head>

Visualiza los resultados de tus Barridos de W&B con la interfaz de usuario de la aplicación W&B. Navega a la interfaz de usuario de la aplicación W&B en [https://wandb.ai/home](https://wandb.ai/home). Elige el proyecto que especificaste cuando inicializaste un Barrido de W&B. Serás redirigido a tu [espacio de trabajo](../app/pages/workspaces.md) del proyecto. Selecciona el **icono de Barrido** en el panel izquierdo (icono de escoba). Desde la [interfaz de usuario de Barrido](./visualize-sweep-results.md), selecciona el nombre de tu Barrido de la lista.

Por defecto, W&B automáticamente creará un gráfico de coordenadas paralelas, un gráfico de importancia de parámetros y un gráfico de dispersión cuando inicies un trabajo de Barrido de W&B.

![Animación que muestra cómo navegar a la interfaz de usuario de Barrido y ver gráficos autogenerados.](/images/sweeps/navigation_sweeps_ui.gif)

Los gráficos de coordenadas paralelas resumen la relación entre un gran número de hiperparámetros y métricas del modelo de un vistazo. Para más información sobre los gráficos de coordenadas paralelas, consulta [Coordenadas Paralelas](../app/features/panels/parallel-coordinates.md).

![Ejemplo de gráfico de coordenadas paralelas.](/images/sweeps/example_parallel_coordiantes_plot.png)

El gráfico de dispersión (izquierda) compara los Runs de W&B que se generaron durante el Barrido. Para más información sobre los gráficos de dispersión, consulta [Gráficos de Dispersión](../app/features/panels/scatter-plot.md).

El gráfico de importancia de parámetros (derecha) enumera los hiperparámetros que fueron los mejores predictores y estuvieron altamente correlacionados con valores deseables de tus métricas. Para más información sobre los gráficos de importancia de parámetros, consulta [Importancia de Parámetros](../app/features/panels/parameter-importance.md).

![Ejemplo de gráfico de dispersión (izquierda) e importancia de parámetros (derecha).](/images/sweeps/scatter_and_parameter_importance.png)

Puedes alterar los valores dependientes e independientes (eje x y eje y) que se usan automáticamente. Dentro de cada panel hay un icono de lápiz llamado **Editar panel**. Selecciona **Editar panel**. Aparecerá un modelo. Dentro del modal, puedes alterar el comportamiento del gráfico.

Para más información sobre todas las opciones de visualización predeterminadas de W&B, consulta [Paneles](../app/features/panels/intro.md). Consulta la documentación de [Visualización de Datos](../tables/intro.md) para información sobre cómo crear gráficos a partir de Runs de W&B que no forman parte de un Barrido de W&B.