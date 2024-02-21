---
description: Delete artifacts interactively with the App UI or programmatically with
  the W&B SDK/
displayed_sidebar: default
---

# Eliminar artefactos

<head>
  <title>Eliminar artefactos de W&B</title>
</head>

Elimina artefactos de manera interactiva con la interfaz de usuario de la aplicación o programáticamente con el SDK de W&B. Cuando eliminas un artefacto, W&B marca ese artefacto como un *borrado suave*. En otras palabras, el artefacto se marca para su eliminación, pero los archivos no se eliminan inmediatamente del almacenamiento.

El contenido del artefacto permanece como un borrado suave, o estado pendiente de eliminación, hasta que un proceso de recolección de basura, que se ejecuta regularmente, revisa todos los artefactos marcados para su eliminación. El proceso de recolección de basura elimina los archivos asociados del almacenamiento si el artefacto y sus archivos asociados no son utilizados por versiones anteriores o posteriores del artefacto.

Las secciones en esta página describen cómo eliminar versiones específicas de artefactos, cómo eliminar una colección de artefactos, cómo eliminar artefactos con y sin alias, y más. Puedes programar cuándo se eliminan los artefactos de W&B con políticas de TTL. Para más información, consulta [Gestionar la retención de datos con la política de TTL de Artefactos](./ttl.md).

:::note
Los artefactos que están programados para eliminarse con una política de TTL, eliminados con el SDK de W&B, o eliminados con la interfaz de usuario de la aplicación de W&B, primero se borran suavemente. Los artefactos que se borran suavemente pasan por la recolección de basura antes de ser eliminados definitivamente.
:::

### Eliminar una versión de artefacto

Para eliminar una versión de artefacto:

1. Selecciona el nombre del artefacto. Esto expandirá la vista del artefacto y listará todas las versiones del artefacto asociadas con ese artefacto.
2. De la lista de artefactos, selecciona la versión del artefacto que quieres eliminar.
3. En el lado derecho del espacio de trabajo, selecciona el menú desplegable kebab.
4. Elige Eliminar.

Una versión de artefacto también se puede eliminar programáticamente mediante el método [delete()](https://docs.wandb.ai/ref/python/artifact#delete). Consulta los ejemplos a continuación.

### Eliminar múltiples versiones de artefactos con alias

El siguiente ejemplo de código demuestra cómo eliminar artefactos que tienen alias asociados. Proporciona la entidad, el nombre del proyecto y el ID de ejecución que creó los artefactos.

```python
import wandb

run = api.run("entidad/proyecto/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

Establece el parámetro `delete_aliases` al valor booleano, `True` para eliminar alias si el artefacto tiene uno o más alias.

```python
import wandb

run = api.run("entidad/proyecto/run_id")

for artifact in run.logged_artifacts():
    # Establece delete_aliases=True para eliminar
    # artefactos con uno o más alias
    artifact.delete(delete_aliases=True)
```

### Eliminar múltiples versiones de artefactos con un alias específico

El código siguiente demuestra cómo eliminar múltiples versiones de artefactos que tienen un alias específico. Proporciona la entidad, el nombre del proyecto y el ID de ejecución que creó los artefactos. Reemplaza la lógica de eliminación por la tuya:

```python
import wandb

runs = api.run("entidad/nombre_del_proyecto/run_id")

# Elimina artefacto con alias 'v3' y 'v4'
for artifact_version in runs.logged_artifacts():
    # Reemplaza con tu propia lógica de eliminación.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### Eliminar todas las versiones de un artefacto que no tienen un alias

El siguiente fragmento de código demuestra cómo eliminar todas las versiones de un artefacto que no tienen un alias. Proporciona el nombre del proyecto y la entidad para las claves `project` y `entity` en `wandb.Api`, respectivamente. Reemplaza los `<>` con el nombre de tu artefacto:

```python
import wandb

# Proporciona tu entidad y un nombre de proyecto cuando
# uses métodos de wandb.Api.
api = wandb.Api(overrides={"project": "proyecto", "entity": "entidad"})

artifact_type, artifact_name = "<>"  # proporciona tipo y nombre
for v in api.artifact_versions(artifact_type, artifact_name):
    # Limpia las versiones que no tienen un alias como 'latest'.
    # NOTA: Puedes poner cualquier lógica de eliminación que quieras aquí.
    if len(v.aliases) == 0:
        v.delete()
```

### Eliminar una colección de artefactos

Para eliminar una colección de artefactos:

1. Navega a la colección de artefactos que deseas eliminar y pasa el cursor sobre ella.
3. Selecciona el menú desplegable kebab junto al nombre de la colección de artefactos.
4. Elige Eliminar.

También puedes eliminar una colección de artefactos programáticamente con el método [delete()](../../ref/python/artifact.md#delete). Proporciona el nombre del proyecto y la entidad para las claves `project` y `entity` en `wandb.Api`, respectivamente:

```python
import wandb

# Proporciona tu entidad y un nombre de proyecto cuando
# uses métodos de wandb.Api.
api = wandb.Api(overrides={"project": "proyecto", "entity": "entidad"})
collection = api.artifact_collection("<tipo_de_artefacto>", "entidad/proyecto/nombre_de_coleccion_de_artefactos")
collection.delete()
```

## Cómo habilitar la recolección de basura según cómo se hospeda W&B
La recolección de basura está habilitada por defecto si usas la nube compartida de W&B. Según cómo hospedes W&B, podrías necesitar tomar pasos adicionales para habilitar la recolección de basura, esto incluye:


* Establecer la variable de entorno `GORILLA_ARTIFACT_GC_ENABLED` en true: `GORILLA_ARTIFACT_GC_ENABLED=true`
* Habilitar el versionamiento de buckets si usas [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) o cualquier otro proveedor de almacenamiento como [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning). Si usas Azure, [habilita la eliminación suave](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview).
  :::note
  La eliminación suave en Azure es equivalente al versionamiento de buckets en otros proveedores de almacenamiento.
  :::

La siguiente tabla describe cómo satisfacer los requisitos para habilitar la recolección de basura basada en tu tipo de despliegue.

La `X` indica que debes satisfacer el requisito:

|                                                | Variable de entorno    | Habilitar versionamiento | 
| -----------------------------------------------| ------------------------| ----------------- | 
| Nube compartida                                |                         |                   | 
| Nube compartida con [conector de almacenamiento seguro](../hosting/secure-storage-connector.md)|                         | X                 | 
| Nube dedicada                                  |                         |                   | 
| Nube dedicada con [conector de almacenamiento seguro](../hosting/secure-storage-connector.md)|                         | X                 | 
| Nube gestionada por el cliente                 | X                       | X                 | 
| On-prem gestionado por el cliente              | X                       | X                 |
 


:::note
El conector de almacenamiento seguro está actualmente disponible solo para Google Cloud Platform y Amazon Web Services.
:::