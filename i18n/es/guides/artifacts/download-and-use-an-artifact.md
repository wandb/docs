---
description: Download and use Artifacts from multiple projects.
displayed_sidebar: default
---

# Descargar y utilizar artefactos

<head>
  <title>Descargar y utilizar artefactos</title>
</head>

Descarga y utiliza un artefacto que ya está almacenado en el servidor de W&B o construye un objeto artefacto y pásalo para que sea deduplicado según sea necesario.

:::note
Los miembros del equipo con asientos solo para visualización no pueden descargar artefactos.
:::

### Descargar y utilizar un artefacto almacenado en W&B

Descarga y utiliza un artefacto que está almacenado en W&B ya sea dentro o fuera de un Run de W&B. Utiliza la API Pública ([`wandb.Api`](../../ref/python/public-api/api.md)) para exportar (o actualizar datos) ya guardados en W&B. Para más información, consulta la [guía de referencia de la API Pública de W&B](../../ref/python/public-api/README.md).

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="insiderun"
  values={[
    {label: 'Durante un run', value: 'insiderun'},
    {label: 'Fuera de un run', value: 'outsiderun'},
    {label: 'CLI de wandb', value: 'cli'},
  ]}>
  <TabItem value="insiderun">

Primero, importa el SDK de Python de W&B. A continuación, crea un [Run](../../ref/python/run.md) de W&B:

```python
import wandb

run = wandb.init(project="<ejemplo>", job_type="<tipo-de-trabajo>")
```

Indica el artefacto que quieres utilizar con el método [`use_artifact`](../../ref/python/run.md#use_artifact). Esto devuelve un objeto run. En el fragmento de código siguiente especificamos un artefacto llamado `'bike-dataset'` con alias `'latest'`:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

Utiliza el objeto devuelto para descargar todo el contenido del artefacto:

```python
datadir = artifact.download()
```

Opcionalmente, puedes pasar una ruta al parámetro root para descargar el contenido del artefacto en un directorio específico. Para más información, consulta la [Guía de Referencia del SDK de Python](../../ref/python/artifact.md#download).

Utiliza el método [`get_path`](../../ref/python/artifact.md#get_path) para descargar solo un subconjunto de archivos:

```python
path = artifact.get_path(name)
```

Esto solo busca el archivo en la ruta `name`. Devuelve un objeto `Entry` con los siguientes métodos:

* `Entry.download`: Descarga el archivo del artefacto en la ruta `name`
* `Entry.ref`: Si la entrada se almacenó como una referencia usando `add_reference`, devuelve el URI

Las referencias que tienen esquemas que W&B sabe manejar se pueden descargar igual que los archivos de artefactos. Para más información, consulta [Rastrear archivos externos](../../guides/artifacts/track-external-files.md).
  
  </TabItem>
  <TabItem value="outsiderun">
  
Primero, importa el SDK de W&B. A continuación, crea un artefacto desde la Clase API Pública. Proporciona la entidad, proyecto, artefacto y alias asociado con ese artefacto:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entidad/proyecto/artefacto:alias")
```

Utiliza el objeto devuelto para descargar el contenido del artefacto:

```python
artifact.download()
```

Opcionalmente, puedes pasar una ruta al parámetro `root` para descargar el contenido del artefacto en un directorio específico. Para más información, consulta la [Guía de Referencia de la API](../../ref/python/artifact.md#download).
  
  </TabItem>
  <TabItem value="cli">

Utiliza el comando `wandb artifact get` para descargar un artefacto del servidor de W&B.

```
$ wandb artifact get proyecto/artefacto:alias --root mnist/
```
  </TabItem>
</Tabs>

### Utilizar un artefacto de un proyecto diferente

Especifica el nombre del artefacto junto con el nombre de su proyecto para referenciar un artefacto. También puedes referenciar artefactos entre entidades especificando el nombre del artefacto con el nombre de su entidad.

El siguiente ejemplo de código demuestra cómo consultar un artefacto de otro proyecto como entrada para nuestro run actual de W&B.

```python
import wandb

run = wandb.init(project="<ejemplo>", job_type="<tipo-de-trabajo>")
# Consulta W&B por un artefacto de otro proyecto y márcalo
# como una entrada para este run.
artifact = run.use_artifact("mi-proyecto/artefacto:alias")

# Utiliza un artefacto de otra entidad y márcalo como una entrada
# para este run.
artifact = run.use_artifact("mi-entidad/mi-proyecto/artefacto:alias")
```

### Construir y utilizar un artefacto simultáneamente

Construye y utiliza un artefacto simultáneamente. Crea un objeto artefacto y pásalo a use_artifact. Esto creará un artefacto en W&B si aún no existe. La API [`use_artifact`](../../ref/python/run.md#use_artifact) es idempotente, por lo que puedes llamarla tantas veces como quieras.

```python
import wandb

artifact = wandb.Artifact("modelo de referencia")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

Para más información sobre la construcción de un artefacto, consulta [Construir un artefacto](../../guides/artifacts/construct-an-artifact.md).