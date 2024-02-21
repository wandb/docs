---
description: Create, construct a W&B Artifact. Learn how to add one or more files
  or a URI reference to an Artifact.
displayed_sidebar: default
---

# Construir artefactos

<head>
  <title>Construir Artefactos</title>
</head>

Utiliza el SDK de Python de W&B para construir artefactos a partir de [W&B Runs](../../ref/python/run.md). Puedes agregar [archivos, directorios, URIs y archivos de runs paralelos a los artefactos](#add-files-to-an-artifact). Después de agregar un archivo a un artefacto, guarda el artefacto en el Servidor W&B o [tu propio servidor privado](../hosting/hosting-options/self-managed.md).

Para información sobre cómo hacer seguimiento de archivos externos, como archivos almacenados en Amazon S3, consulta la página [Hacer seguimiento de archivos externos](./track-external-files.md).

## Cómo construir un artefacto

Construye un [Artefacto W&B](../../ref/python/artifact.md) en tres pasos:

### 1. Crea un objeto de artefacto Python con `wandb.Artifact()`

Inicializa la clase [`wandb.Artifact()`](../../ref/python/artifact.md) para crear un objeto de artefacto. Especifica los siguientes parámetros:

* **Nombre**: Especifica un nombre para tu artefacto. El nombre debe ser único, descriptivo y fácil de recordar. Usa el nombre de un artefacto para ambos: identificar el artefacto en la interfaz de usuario de la aplicación W&B y cuando quieras usar ese artefacto.
* **Tipo**: Proporciona un tipo. El tipo debe ser simple, descriptivo y corresponder a un único paso de tu pipeline de aprendizaje automático. Los tipos comunes de artefactos incluyen `'dataset'` o `'model'`.

:::tip
El "nombre" y "tipo" que proporcionas se usan para crear un grafo acíclico dirigido. Esto significa que puedes ver el linaje de un artefacto en la App W&B.

Consulta la [Explorar y recorrer grafos de artefactos](./explore-and-traverse-an-artifact-graph.md) para más información.
:::

:::caution
Los artefactos no pueden tener el mismo nombre, incluso si especificas un tipo diferente para el parámetro de tipos. En otras palabras, no puedes crear un artefacto llamado ‘gatos’ de tipo ‘dataset’ y otro artefacto con el mismo nombre de tipo ‘modelo’.
:::

Opcionalmente, puedes proporcionar una descripción y metadatos cuando inicializas un objeto de artefacto. Para más información sobre los atributos y parámetros disponibles, consulta la definición de la Clase [wandb.Artifact](../../ref/python/artifact.md) en la Guía de Referencia del SDK de Python.

El siguiente ejemplo demuestra cómo crear un artefacto de dataset:

```python
import wandb

artifact = wandb.Artifact(name="<reemplazar>", type="<reemplazar>")
```

Reemplaza los argumentos de cadena en el fragmento de código anterior con tu propio nombre y tipo.

### 2. Agrega uno o más archivos al artefacto

Agrega archivos, directorios, referencias URI externas (como Amazon S3) y más con métodos de artefacto. Por ejemplo, para agregar un único archivo de texto, usa el método [`add_file`](../../ref/python/artifact.md#add_file):

```python
artifact.add_file(local_path="hello_world.txt", name="nombre-opcional")
```

También puedes agregar múltiples archivos con el método [`add_dir`](../../ref/python/artifact.md#add_dir). Para más información sobre cómo agregar archivos, consulta [Actualizar un artefacto](./update-an-artifact.md).

### 3. Guarda tu artefacto en el servidor W&B

Finalmente, guarda tu artefacto en el servidor W&B. Los artefactos están asociados con un run. Por lo tanto, utiliza el método [`log_artifact()`](../../ref/python/run#log\_artifact) de objetos de run para guardar el artefacto.

```python
# Crear un W&B Run. Reemplaza 'tipo-de-trabajo'.
run = wandb.init(project="ejemplo-artefactos", job_type="tipo-de-trabajo")

run.log_artifact(artifact)
```

Opcionalmente, puedes construir un artefacto fuera de un W&B run. Para más información, consulta [Hacer seguimiento de archivos externos](./track-external-files).

:::caution
Las llamadas a `log_artifact` se realizan de forma asincrónica para cargas más eficientes. Esto puede causar comportamientos sorprendentes al registrar artefactos en un bucle. Por ejemplo:

```python
for i in range(10):
    a = wandb.Artifact(
        "carrera",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... añadir archivos al artefacto a ...
    run.log_artifact(a)
```

La versión del artefacto **v0** NO está garantizada para tener un índice de 0 en sus metadatos, ya que los artefactos pueden ser registrados en un orden arbitrario.
:::

## Agregar archivos a un artefacto

Las siguientes secciones demuestran cómo construir artefactos con diferentes tipos de archivos y desde runs paralelos.

Para los siguientes ejemplos, asume que tienes un directorio de proyecto con múltiples archivos y una estructura de directorios:

```
directorio-proyecto
|-- imágenes
|   |-- gato.png
|   +-- perro.png
|-- checkpoints
|   +-- modelo.h5
+-- modelo.h5
```

### Agregar un único archivo

El siguiente fragmento de código demuestra cómo agregar un único archivo local a tu artefacto:

```python
# Agregar un único archivo
artifact.add_file(local_path="ruta/archivo.formato")
```

Por ejemplo, supongamos que tenías un archivo llamado `'archivo.txt'` en tu directorio local de trabajo.

```python
artifact.add_file("ruta/archivo.txt")  # Añadido como `archivo.txt'
```

El artefacto ahora tiene el siguiente contenido:

```
archivo.txt
```

Opcionalmente, pasa la ruta deseada dentro del artefacto para el parámetro `name`.

```python
artifact.add_file(local_path="ruta/archivo.formato", name="nueva/ruta/archivo.formato")
```

El artefacto se almacena como:

```
nueva/ruta/archivo.txt
```

| Llamada API                                                  | Artefacto resultante |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('modelo.h5')`                           | modelo.h5           |
| `artifact.add_file('checkpoints/modelo.h5')`               | modelo.h5           |
| `artifact.add_file('modelo.h5', name='modelos/mimodelo.h5')` | modelos/mimodelo.h5  |

### Agregar múltiples archivos

El siguiente fragmento de código demuestra cómo agregar un directorio local completo a tu artefacto:

```python
# Agregar recursivamente un directorio
artifact.add_dir(local_path="ruta/archivo.formato", name="prefijo-opcional")
```

Las siguientes llamadas API producen el siguiente contenido del artefacto:

| Llamada API                                    | Artefacto resultante                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('imágenes')`                | <p><code>gato.png</code></p><p><code>perro.png</code></p> |
| `artifact.add_dir('imágenes', name='imágenes')` | <p><code>imágenes/gato.png</code></p><p><code>imágenes/perro.png</code></p> |
| `artifact.new_file('hola.txt')`            | `hola.txt`                                            |

### Agregar una referencia URI

Los artefactos hacen seguimiento de checksums y otra información para reproducibilidad si la URI tiene un esquema que la biblioteca W&B sabe cómo manejar.

Agrega una referencia URI externa a un artefacto con el método [`add_reference`](../../ref/python/artifact#add\_reference). Reemplaza la cadena `'uri'` con tu propia URI. Opcionalmente pasa la ruta deseada dentro del artefacto para el parámetro name.

```python
# Agregar una referencia URI
artifact.add_reference(uri="uri", name="nombre-opcional")
```

Actualmente, los artefactos admiten los siguientes esquemas URI:

* `http(s)://`: Una ruta a un archivo accesible a través de HTTP. El artefacto hará seguimiento de checksums en forma de etags y metadatos de tamaño si el servidor HTTP admite las cabeceras de respuesta `ETag` y `Content-Length`.
* `s3://`: Una ruta a un objeto o prefijo de objeto en S3. El artefacto hará seguimiento de checksums e información de versionamiento (si el bucket tiene versionamiento de objetos habilitado) para los objetos referenciados. Los prefijos de objeto se expanden para incluir los objetos bajo el prefijo, hasta un máximo de 10,000 objetos.
* `gs://`: Una ruta a un objeto o prefijo de objeto en GCS. El artefacto hará seguimiento de checksums e información de versionamiento (si el bucket tiene versionamiento de objetos habilitado) para los objetos referenciados. Los prefijos de objeto se expanden para incluir los objetos bajo el prefijo, hasta un máximo de 10,000 objetos.

Las siguientes llamadas API producirán los siguientes artefactos:

| Llamada API                                                                      | Contenido del artefacto resultante                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://mi-bucket/modelo.h5')`                           | `modelo.h5`                                                           |
| `artifact.add_reference('s3://mi-bucket/checkpoints/modelo.h5')`               | `modelo.h5`                                                           |
| `artifact.add_reference('s3://mi-bucket/modelo.h5', name='modelos/mimodelo.h5')` | `modelos/mimodelo.h5`                                                  |
| `artifact.add_reference('s3://mi-bucket/imágenes')`                             | <p><code>gato.png</code></p><p><code>perro.png</code></p>               |
| `artifact.add_reference('s3://mi-bucket/imágenes', name='imágenes')`              | <p><code>imágenes/gato.png</code></p><p><code>imágenes/perro.png</code></p> |

### Agregar archivos a artefactos desde runs paralelos

Para grandes datasets o entrenamiento distribuido, múltiples runs paralelos pueden necesitar contribuir a un único artefacto.

```python
import wandb
import time

# Usaremos ray para lanzar nuestros runs en paralelo
# con fines demostrativos. Puedes orquestar
# tus runs paralelos como quieras.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "artefacto-paralelo"
table_name = "tabla-distribuida"
parts_path = "partes"
num_parallel = 5

# Cada lote de escritores paralelos debe tener su propio
# nombre de grupo único.
group_name = "grupo-escritor-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    Nuestro trabajo de escritura. Cada escritor agregará una imagen al artefacto.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # Agregar datos a una tabla wandb. En este caso usamos datos de ejemplo
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # Agregar la tabla a carpeta en el artefacto
        artifact.add(table, "{}/tabla_{}".format(parts_path, i))

        # Actualizar el artefacto crea o añade datos al artefacto
        run.upsert_artifact(artifact)


# Lanzar tus runs en paralelo
result_ids = [train.remote(i) for i in range(num_parallel)]

# Unirse a todos los escritores para asegurarse de que sus archivos hayan
# sido añadidos antes de terminar el artefacto.
ray.get(result_ids)

# Una vez que todos los escritores hayan terminado, terminar el artefacto
# para marcarlo como listo.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # Crear una "PartitionTable" apuntando a la carpeta de tablas
    # y añadirla al artefacto.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Terminar artefacto finaliza el artefacto, impidiendo futuras "actualizaciones"
    # a esta versión.
    run.finish_artifact(artifact)
```