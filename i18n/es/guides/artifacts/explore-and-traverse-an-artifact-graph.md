---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: default
---

# Explora y atraviesa gráficos de artefactos

<head>
    <title>Explora gráficos acíclicos directos de Artefactos de W&B.</title>
</head>

W&B rastrea automáticamente los artefactos que un run ha registrado así como los artefactos que un run ha utilizado. Explora el linaje de un artefacto con la interfaz de usuario de la aplicación W&B o programáticamente.

## Atraviesa un artefacto con la interfaz de usuario de la aplicación W&B

La vista de gráfico muestra una visión general de tu pipeline.

Para ver un gráfico de artefacto:

1. Navega a tu proyecto en la interfaz de usuario de la aplicación W&B
2. Elige el icono de artefacto en el panel izquierdo.
3. Selecciona **Linaje**.

El `tipo` que proporcionas al crear runs y artefactos se utiliza para crear el gráfico. La entrada y salida de un run o artefacto se representa en el gráfico con flechas. Los artefactos están representados por rectángulos azules y los Runs por rectángulos verdes.

El tipo de artefacto que proporcionas se encuentra en el encabezado azul oscuro al lado de la etiqueta **ARTEFACTO**. El nombre del artefacto, junto con la versión del artefacto, se muestra en la región azul claro debajo de la etiqueta **ARTEFACTO**.

El tipo de trabajo que proporcionas al inicializar un run se encuentra al lado de la etiqueta **RUN**. El nombre del run de W&B se encuentra en la región verde claro debajo de la etiqueta **RUN**.

:::info
Puedes ver el tipo y el nombre de los artefactos tanto en la barra lateral izquierda como en la pestaña **Linaje**. 
:::

Por ejemplo, en la imagen siguiente, un artefacto fue definido con un tipo llamado "raw_dataset" (cuadrado rosa). El nombre del artefacto se llama "MNIST_raw" (línea rosa). El artefacto fue luego utilizado para entrenamiento. El nombre del run de entrenamiento se llama "vivid-snow-42". Ese run luego produjo un artefacto "modelo" (cuadrado naranja) llamado "mnist-19pofeku".

![Vista DAG de artefactos, runs utilizados para un experimento.](/images/artifacts/example_dag_with_sidebar.png)

Para una vista más detallada, selecciona el alternador **Explode** en la parte superior izquierda del panel de control. El gráfico expandido muestra detalles de cada run y cada artefacto en el proyecto que fue registrado. Pruébalo tú mismo en esta [página de ejemplo de Graph](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/v0/lineage).

## Atraviesa un artefacto programáticamente

Crea un objeto artefacto con la API Pública de W&B ([wandb.Api](../../ref/python/public-api/api.md)). Proporciona el nombre del proyecto, artefacto y alias del artefacto:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("project/artifact:alias")
```

Usa los métodos [`logged_by`](../../ref/python/artifact.md#logged_by) y [`used_by`](../../ref/python/artifact.md#used_by) del objeto artefacto para caminar por el gráfico desde el artefacto:

```python
# Camina hacia arriba y hacia abajo en el gráfico desde un artefacto:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

#### Atraviesa desde un run

Crea un objeto artefacto con la API Pública de W&B ([wandb.Api.Run](../../ref/python/public-api/run.md)). Proporciona el nombre de la entidad, proyecto y ID del run:

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
```

Usa los métodos [`logged_artifacts`](../../ref/python/public-api/run.md#logged_artifacts) y [`used_artifacts`](../../ref/python/public-api/run.md#used_artifacts) para caminar por el gráfico desde un run dado:

```python
# Camina hacia arriba y hacia abajo en el gráfico desde un run:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```