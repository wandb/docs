---
description: Some features on this page are in beta, hidden behind a feature flag.
  Add `weave-plot` to your bio on your profile page to unlock all related features.
slug: /guides/app/features/panels/weave
displayed_sidebar: default
---

# Weave

## Introducción

Los Paneles Weave permiten a los usuarios consultar directamente a W&B por datos, visualizar los resultados y analizarlos de manera interactiva.

![](/images/weave/pretty_panel.png)

:::tip
Vea [este reporte](http://wandb.me/keras-xla-benchmark) para ver cómo este equipo utilizó los Paneles Weave para visualizar sus benchmarks.
:::

## Crear un Panel Weave

Para agregar un Panel Weave:

* En tu Espacio de Trabajo, haz clic en `Agregar Panel` y selecciona `Weave`.
![](/images/weave/add_weave_panel_workspace.png)
* En un Reporte:
  * Escribe `/weave` y selecciona `Weave` para agregar un Panel Weave independiente.
  ![](/images/weave/add_weave_panel_report_1.png)
  * Escribe `/Panel grid` -> `Panel grid` y luego haz clic en `Agregar panel` -> `Weave` para agregar un Panel Weave asociado con un conjunto de runs.
  ![](/images/weave/add_weave_panel_report_2.png)

## Componentes

### Expresión Weave

Las expresiones Weave permiten al usuario consultar los datos almacenados en W&B - desde runs, hasta artefactos, modelos, tablas y más. Expresión weave común que puedes generar cuando registras una Tabla con `wandb.log({"cifar10_sample_table":<MY_TABLE>})`:

![](/images/weave/basic_weave_expression.png)

Desglosemos esto:

* `runs` es una variable automáticamente inyectada en las Expresiones del Panel Weave cuando el Panel Weave está en un Espacio de Trabajo. Su "valor" es la lista de runs que son visibles para ese Espacio de Trabajo en particular. [Lee sobre los diferentes atributos disponibles dentro de un run aquí](../../../../track/public-api-guide.md#understanding-the-different-attributes).
* `summary` es una operación que devuelve el objeto Resumen de un Run. Nota: las operaciones son "mapeadas", lo que significa que esta operación se aplica a cada Run en la lista, resultando en una lista de objetos Resumen.
* `["cifar10_sample_table"]` es una operación de Selección (denotada con corchetes), con un parámetro de "predictions". Dado que los objetos Resumen actúan como diccionarios o mapas, esta operación "selecciona" el campo "predictions" de cada objeto Resumen.

Para aprender cómo escribir tus propias consultas de manera interactiva, consulta [este reporte](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr), que va desde las operaciones básicas disponibles en Weave hasta otras visualizaciones avanzadas de tus datos.

### Configuración de Weave

Selecciona el icono de engranaje en la esquina superior izquierda del panel para expandir la configuración de Weave. Esto permite al usuario configurar el tipo de panel y los parámetros para el panel de resultado.

![](/images/weave/weave_panel_config.png)

### Panel de resultado de Weave

Finalmente, el panel de resultado de Weave muestra el resultado de la expresión Weave, utilizando el panel de Weave seleccionado, configurado por la configuración para mostrar los datos de forma interactiva. Las siguientes imágenes muestran una Tabla y un Gráfico de los mismos datos.

![](/images/weave/result_panel_table.png)

![](/images/weave/result_panel_plot.png)

## Operaciones básicas

### Ordenar
Puedes ordenar fácilmente desde las opciones de la columna
![](/images/weave/weave_sort.png)

### Filtrar
Puedes filtrar directamente en la consulta o usando el botón de filtro en la esquina superior izquierda (segunda imagen)
![](/images/weave/weave_filter_1.png)
![](/images/weave/weave_filter_2.png)

### Mapear
Las operaciones de mapeo iteran sobre listas y aplican una función a cada elemento en los datos. Puedes hacer esto directamente con una consulta Weave o insertando una nueva columna desde las opciones de la columna.
![](/images/weave/weave_map.png)
![](/images/weave/weave_map.gif)

### Agrupar por
Puedes agrupar usando una consulta o desde las opciones de la columna.
![](/images/weave/weave_groupby.png)
![](/images/weave/weave_groupby.gif)

### Concatenar
La operación de concatenación te permite concatenar 2 tablas y concatenar o unir desde la configuración del panel
![](/images/weave/weave_concat.gif)

### Unir
También es posible unir tablas directamente en la consulta, donde:
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat` es la primera tabla
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat` es la segunda tabla
* `(row) => row["Label"]` son selectores para cada tabla, determinando sobre qué columna unir
* `"Table1"` y `"Table2"` son los nombres de cada tabla al unirse
* `true` y `false` son para configuraciones de unión interna/externa izquierda y derecha
![](/images/weave/weave_join.png)

## Objeto Runs
Entre otras cosas, Weave te permite acceder al objeto `runs`, que almacena un registro detallado de tus experimentos. Puedes encontrar más detalles sobre esto en [esta sección](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) del reporte pero, como vista rápida, el objeto `runs` tiene disponible:
* `summary`: Un diccionario de información que resume los resultados del run. Esto puede ser escalares como precisión y pérdida, o archivos grandes. Por defecto, `wandb.log()` establece el resumen al valor final de una serie de tiempo registrada. Puedes establecer el contenido del resumen directamente. Piensa en el resumen como las salidas del run.
* `history`: Una lista de diccionarios destinada a almacenar valores que cambian mientras el modelo se está entrenando, como la pérdida. El comando `wandb.log()` se añade a este objeto.
* `config`: Un diccionario de la información de configuración del run, como los hiperparámetros para un run de entrenamiento o los métodos de preprocesamiento para un run que crea un Artefacto de dataset. Piensa en estos como los "inputs" del run.
![](/images/weave/weave_runs_object.png)

## Acceso a Artefactos

Los Artefactos son un concepto central en W&B. Son una colección nombrada y versionada de archivos y directorios. Usa Artefactos para rastrear pesos de modelos, datasets, y cualquier otro archivo o directorio. Los Artefactos se almacenan en W&B y pueden ser descargados o usados en otros runs. Puedes encontrar más detalles y ejemplos en [esta sección](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) del reporte. Los Artefactos normalmente se acceden desde el objeto `project`:
* `project.artifactVersion()`: devuelve la versión específica del artefacto para un nombre y versión dados dentro de un proyecto
* `project.artifact("")`: devuelve el artefacto para un nombre dado dentro de un proyecto. Luego puedes usar `.versions` para obtener una lista de todas las versiones de este artefacto
* `project.artifactType()`: devuelve el `artifactType` para un nombre dado dentro de un proyecto. Luego puedes usar `.artifacts` para obtener una lista de todos los artefactos con este tipo
* `project.artifactTypes`: devuelve una lista de todos los tipos de artefactos bajo el proyecto
![](/images/weave/weave_artifacts.png)