---
description: Update an existing Artifact inside and outside of a W&B Run.
displayed_sidebar: default
---

# Actualizar artefactos

<head>
  <title>Actualizar artefactos</title>
</head>

Pasa los valores deseados para actualizar la `descripción`, `metadatos` y `alias` de un artefacto. Llama al método `save()` para actualizar el artefacto en los servidores de W&B. Puedes actualizar un artefacto durante un Run de W&B o fuera de un Run.

Usa la API Pública de W&B ([`wandb.Api`](../../ref/python/public-api/api.md)) para actualizar un artefacto fuera de un run. Usa la API de Artefactos ([`wandb.Artifact`](../../ref/python/artifact.md)) para actualizar un artefacto durante un run.

:::caution
No puedes actualizar el alias de un artefacto que está vinculado a un modelo en el Registro de Modelos.
:::


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'Durante un Run', value: 'duringrun'},
    {label: 'Fuera de un Run', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

El siguiente ejemplo de código demuestra cómo actualizar la descripción de un artefacto usando la API [`wandb.Artifact`](../../ref/python/artifact.md):

```python
import wandb

run = wandb.init(project="<ejemplo>", job_type="<tipo-de-trabajo>")
artifact = run.use_artifact("<nombre-del-artefacto>:<alias>")

artifact = wandb.Artifact("")
run.use_artifact(artifact)
artifact.description = "<descripción>"
artifact.save()
```
  </TabItem>
  <TabItem value="outsiderun">

El siguiente ejemplo de código demuestra cómo actualizar la descripción de un artefacto usando la API `wandb.Api`:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entidad/proyecto/artefacto:alias")

# Actualizar la descripción
artifact.description = "Mi nueva descripción"

# Actualizar selectivamente las claves de metadatos
artifact.metadata["oldKey"] = "nuevo valor"

# Reemplazar completamente los metadatos
artifact.metadata = {"newKey": "nuevo valor"}

# Añadir un alias
artifact.aliases.append("mejor")

# Eliminar un alias
artifact.aliases.remove("último")

# Reemplazar completamente los alias
artifact.aliases = ["reemplazado"]

# Persistir todas las modificaciones del artefacto
artifact.save()
```

Para más información, consulta la [API de Artefactos](../../ref/python/artifact.md) de Weights and Biases.
  </TabItem>
</Tabs>