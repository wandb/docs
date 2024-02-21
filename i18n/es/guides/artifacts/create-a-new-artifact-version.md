---
description: Create a new artifact version from a single run or from a distributed
  process.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Crear nuevas versiones de artefactos

<head>
    <title>Crear nuevas versiones de artefactos desde runs individuales y multiproceso.</title>
</head>

Crea una nueva versión de artefacto con un [run](../runs/intro.md) individual o colaborativamente con runs distribuidos. Opcionalmente, puedes crear una nueva versión de artefacto a partir de una versión anterior conocida como un [artefacto incremental](#create-a-new-artifact-version-from-an-existing-version).

:::tip
Recomendamos que crees un artefacto incremental cuando necesites aplicar cambios a un subconjunto de archivos en un artefacto, donde el tamaño del artefacto original es significativamente mayor.
:::


<!-- ![Diagrama de visión general de artefacto](/images/artifacts/incremental_artifacts_Diagram.png) -->

## Crear nuevas versiones de artefactos desde cero
Hay dos maneras de crear una nueva versión de artefacto: desde un run individual y desde runs distribuidos. Se definen de la siguiente manera:


* **Run individual**: Un run individual proporciona todos los datos para una nueva versión. Este es el caso más común y es el más adecuado cuando el run recrea completamente los datos necesarios. Por ejemplo: generar modelos guardados o predicciones de modelos en una tabla para análisis.
* **Runs distribuidos**: Un conjunto de runs proporciona colectivamente todos los datos para una nueva versión. Esto es más adecuado para trabajos distribuidos que tienen múltiples runs generando datos, a menudo en paralelo. Por ejemplo: evaluar un modelo de manera distribuida y generar las predicciones.


W&B creará un nuevo artefacto y le asignará un alias `v0` si pasas un nombre al API `wandb.Artifact` que no existe en tu proyecto. W&B verifica los contenidos cuando vuelves a registrar en el mismo artefacto. Si el artefacto cambió, W&B guarda una nueva versión `v1`.  

W&B recuperará un artefacto existente si pasas un nombre y tipo de artefacto al API `wandb.Artifact` que coincide con un artefacto existente en tu proyecto. El artefacto recuperado tendrá una versión mayor que 1. 

![](/images/artifacts/single_distributed_artifacts.png)

### Run individual
Registra una nueva versión de un Artefacto con un run individual que produce todos los archivos en el artefacto. Este caso ocurre cuando un run individual produce todos los archivos en el artefacto. 

Según tu caso de uso, selecciona una de las pestañas a continuación para crear una nueva versión de artefacto dentro o fuera de un run:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Dentro de un run', value: 'inside'},
    {label: 'Fuera de un run', value: 'outside'},
  ]}>
  <TabItem value="inside">

Crea una versión de artefacto dentro de un run de W&B:

1. Crea un run con `wandb.init`. (Línea 1)
2. Crea un nuevo artefacto o recupera uno existente con `wandb.Artifact`. (Línea 2)
3. Agrega archivos al artefacto con `.add_file`. (Línea 9)
4. Registra el artefacto en el run con `.log_artifact`. (Línea 10)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # Agrega archivos y activos al artefacto usando
    # `.add`, `.add_file`, `.add_dir`, y `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

Crea una versión de artefacto fuera de un run de W&B:

1. Crea un nuevo artefacto o recupera uno existente con `wanb.Artifact`. (Línea 1)
2. Agrega archivos al artefacto con `.add_file`. (Línea 4)
3. Guarda el artefacto con `.save`. (Línea 5)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# Agrega archivos y activos al artefacto usando
# `.add`, `.add_file`, `.add_dir`, y `.add_reference`
artifact.add_file("image1.png")
artifact.save()
```  
  </TabItem>
</Tabs>

### Runs distribuidos

Permite que una colección de runs colabore en una versión antes de comprometerla. Esto es en contraste con el modo de run individual descrito anteriormente donde un run proporciona todos los datos para una nueva versión.


:::info
1. Cada run en la colección necesita ser consciente del mismo ID único (llamado `distributed_id`) para colaborar en la misma versión. Por defecto, si está presente, W&B usa el `group` del run como se establece por `wandb.init(group=GROUP)` como el `distributed_id`.
2. Debe haber un run final que "comprometa" la versión, bloqueando permanentemente su estado.
3. Usa `upsert_artifact` para agregar al artefacto colaborativo y `finish_artifact` para finalizar el compromiso.
:::

Considera el siguiente ejemplo. Diferentes runs (etiquetados a continuación como **Run 1**, **Run 2**, y **Run 3**) agregan un archivo de imagen diferente al mismo artefacto con `upsert_artifact`.

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Agrega archivos y activos al artefacto usando
    # `.add`, `.add_file`, `.add_dir`, y `.add_reference`
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Agrega archivos y activos al artefacto usando
    # `.add`, `.add_file`, `.add_dir`, y `.add_reference`
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Debe ejecutarse después de que el Run 1 y Run 2 se completen. El Run que llama a `finish_artifact` puede incluir archivos en el artefacto, pero no necesita hacerlo.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Agrega archivos y activos al artefacto
    # `.add`, `.add_file`, `.add_dir`, y `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## Crear una nueva versión de artefacto a partir de una versión existente

Agrega, modifica o elimina un subconjunto de archivos de una versión de artefacto anterior sin la necesidad de volver a indexar los archivos que no cambiaron. Agregar, modificar o eliminar un subconjunto de archivos de una versión de artefacto anterior crea una nueva versión de artefacto conocida como un *artefacto incremental*.

![](/images/artifacts/incremental_artifacts.png)

Aquí hay algunos escenarios para cada tipo de cambio incremental que podrías encontrar:

- añadir: periódicamente agregas un nuevo subconjunto de archivos a un dataset después de recolectar un nuevo lote.
- eliminar: descubriste varios archivos duplicados y quieres eliminarlos de tu artefacto.
- actualizar: corregiste anotaciones para un subconjunto de archivos y quieres reemplazar los archivos antiguos con los correctos.

Podrías crear un artefacto desde cero para realizar la misma función que un artefacto incremental. Sin embargo, cuando creas un artefacto desde cero, necesitarás tener todos los contenidos de tu artefacto en tu disco local. Al realizar un cambio incremental, puedes agregar, eliminar o modificar un solo archivo sin cambiar los archivos de una versión de artefacto anterior.


:::info
Puedes crear un artefacto incremental dentro de un run individual o con un conjunto de runs (modo distribuido).
:::


Sigue el procedimiento a continuación para cambiar incrementalmente un artefacto:

1. Obtén la versión de artefacto en la que quieres realizar un cambio incremental:

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Dentro de un run', value: 'inside'},
    {label: 'Fuera de un run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

  </TabItem>
</Tabs>





2. Crea un borrador con:

```python
draft_artifact = saved_artifact.new_draft()
```

3. Realiza cualquier cambio incremental que quieras ver en la próxima versión. Puedes agregar, eliminar o modificar una entrada existente.

Selecciona una de las pestañas para un ejemplo de cómo realizar cada uno de estos cambios:


<Tabs
  defaultValue="add"
  values={[
    {label: 'Añadir', value: 'add'},
    {label: 'Eliminar', value: 'remove'},
    {label: 'Modificar', value: 'modify'},
  ]}>
  <TabItem value="add">

Agrega un archivo a una versión de artefacto existente con el método `add_file`:

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
También puedes agregar múltiples archivos agregando un directorio con el método `add_dir`.
:::

  </TabItem>
  <TabItem value="remove">

Elimina un archivo de una versión de artefacto existente con el método `remove`:

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
También puedes eliminar múltiples archivos con el método `remove` pasando una ruta de directorio.
:::

  </TabItem>
  <TabItem value="modify">

Modifica o reemplaza contenidos eliminando los contenidos antiguos del borrador y agregando de nuevo los nuevos contenidos:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

<!-- :::tip
El método para agregar o modificar un artefacto son los mismos. Las entradas se reemplazan (en lugar de duplicarse), cuando pasas un nombre de archivo para una entrada que ya existe.
::: -->

4. Por último, registra o guarda tus cambios. Las siguientes pestañas te muestran cómo guardar tus cambios dentro y fuera de un run de W&B. Selecciona la pestaña que sea apropiada para tu caso de uso:


<Tabs
  defaultValue="inside"
  values={[
    {label: 'Dentro de un run', value: 'inside'},
    {label: 'Fuera de un run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
run.log_artifact(draft_artifact)
```

  </TabItem>
  <TabItem value="outside">


```python
draft_artifact.save()
```

  </TabItem>
</Tabs>


Poniéndolo todo junto, los ejemplos de código anteriores se ven así: 

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Dentro de un run', value: 'inside'},
    {label: 'Fuera de un run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # busca el artefacto y lo ingresa en tu run
    draft_artifact = saved_artifact.new_draft()  # crea una versión borrador

    # modifica un subconjunto de archivos en la versión borrador
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # registra tus cambios para crear una nueva versión y márcalo como salida para tu run
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # carga tu artefacto
draft_artifact = saved_artifact.new_draft()  # crea una versión borrador

# modifica un subconjunto de archivos en la versión borrador
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # compromete los cambios al borrador
```

  </TabItem>
</Tabs>