---
description: Answers to frequently asked question about W&B Artifacts.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Preguntas frecuentes sobre Artefactos

<head>
  <title>Preguntas Frecuentes Sobre Artefactos</title>
</head>

Las siguientes preguntas son las más comunes acerca de [Artefactos de W&B](#questions-about-artifacts) y [Flujos de trabajo de Artefactos de W&B](#questions-about-artifacts-workflows).

## Preguntas sobre Artefactos

### ¿Quién tiene acceso a mis artefactos?

Los artefactos heredan el acceso de su proyecto padre:

* Si el proyecto es privado, entonces solo los miembros del equipo del proyecto tienen acceso a sus artefactos.
* Para proyectos públicos, todos los usuarios tienen acceso de lectura a los artefactos pero solo los miembros del equipo del proyecto pueden crearlos o modificarlos.
* Para proyectos abiertos, todos los usuarios tienen acceso de lectura y escritura a los artefactos.

## Preguntas sobre Flujos de trabajo de Artefactos

Esta sección describe los flujos de trabajo para gestionar y editar Artefactos. Muchos de estos flujos de trabajo utilizan [la API de W&B](../track/public-api-guide.md), el componente de [nuestra biblioteca cliente](../../ref/python/README.md) que proporciona acceso a los datos almacenados con W&B.

### ¿Cómo registro un artefacto en un run existente?

Ocasionalmente, puedes querer marcar un artefacto como el resultado de un run previamente registrado. En ese escenario, puedes [reinicializar el run antiguo](../runs/resuming.md) y registrar nuevos artefactos en él de la siguiente manera:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### ¿Cómo establezco una política de retención o expiración en mi artefacto?

Si tienes artefactos que están sujetos a regulaciones de privacidad de datos como artefactos de dataset que contienen PII, o quieres programar la eliminación de una versión de un artefacto para gestionar tu almacenamiento, puedes establecer una política TTL (time-to-live). Aprende más en [esta](./ttl.md) guía.

### ¿Cómo puedo encontrar los artefactos registrados o consumidos por un run? ¿Cómo puedo encontrar los runs que produjeron o consumieron un artefacto?

W&B rastrea automáticamente los artefactos que un run dado ha registrado así como los artefactos que un run dado ha utilizado y usa la información para construir un grafo de artefacto -- un grafo bipartito, dirigido, acíclico cuyos nodos son runs y artefactos, como [este](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) (haz clic en "Explode" para ver el grafo completo).

Puedes navegar este grafo programáticamente con [la API Pública](../../ref/python/public-api/README.md), comenzando desde un run o un artefacto.

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: 'Desde un Artefacto', value: 'from_artifact'},
    {label: 'Desde un Run', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# Subir por el grafo desde un artefacto:
producer_run = artifact.logged_by()
# Bajar por el grafo desde un artefacto:
consumer_runs = artifact.used_by()

# Bajar por el grafo desde un run:
next_artifacts = consumer_runs[0].logged_artifacts()
# Subir por el grafo desde un run:
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# Bajar por el grafo desde un run:
produced_artifacts = run.logged_artifacts()
# Subir por el grafo desde un run:
consumed_artifacts = run.used_artifacts()

# Subir por el grafo desde un artefacto:
earlier_run = consumed_artifacts[0].logged_by()
# Bajar por el grafo desde un artefacto:
consumer_runs = produced_artifacts[0].used_by()
```

  </TabItem>
</Tabs>

### ¿Cómo es mejor registrar modelos de runs en un barrido?

Un patrón efectivo para registrar modelos en un [barrido](../sweeps/intro.md) es tener un artefacto de modelo para el barrido, donde las versiones corresponderán a diferentes runs del barrido. Más concretamente, harías:

```python
wandb.Artifact(name="sweep_name", type="model")
```

### ¿Cómo encuentro un artefacto del mejor run en un barrido?

Puedes usar el siguiente código para recuperar los artefactos asociados con el run de mejor desempeño en un barrido:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```

### ¿Cómo guardo código?‌

Usa `save_code=True` en `wandb.init` para guardar el script principal o notebook donde estás lanzando el run. Para guardar todo tu código en un run, versiona el código con Artefactos. He aquí un ejemplo:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```

### ¿Usando artefactos con múltiples arquitecturas y runs?

Hay muchas maneras en las que puedes pensar en _versionar_ un modelo. Artefactos te proporciona una herramienta para implementar el versionamiento de modelos como mejor te parezca. Un patrón común para proyectos que exploran múltiples arquitecturas de modelos a lo largo de varios runs es separar los artefactos por arquitectura. Como ejemplo, uno podría hacer lo siguiente:

1. Crear un nuevo artefacto para cada arquitectura de modelo diferente. Puedes usar el atributo `metadata` de los artefactos para describir la arquitectura con más detalle (similar a cómo usarías `config` para un run).
2. Para cada modelo, registrar periódicamente checkpoints con `log_artifact`. W&B construirá automáticamente un historial de esos checkpoints, anotando el checkpoint más reciente con el alias `latest` para que puedas referirte al último checkpoint de cualquier arquitectura de modelo usando `nombre-arquitectura:latest`

## Referencia de Preguntas Frecuentes sobre Artefactos

### ¿Cómo puedo obtener estos IDs de Versión y ETags en W&B?

Si has registrado una referencia de artefacto con W&B y si el versionamiento está habilitado en tus buckets entonces los IDs de versión se pueden ver en la UI de S3. Para obtener estos IDs de versión y ETags en W&B, puedes obtener el artefacto y luego obtener las correspondientes entradas del manifiesto. Por ejemplo:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```