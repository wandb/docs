---
description: Track files saved outside the W&B such as in an Amazon S3 bucket, GCS
  bucket, HTTP file server, or even an NFS share.
displayed_sidebar: default
---

# Seguimiento de archivos externos

<head>
	<title>Seguimiento de archivos externos con artefactos de referencia</title>
</head>

Utiliza **artefactos de referencia** para hacer seguimiento de archivos guardados fuera del sistema W&B, por ejemplo, en un bucket de Amazon S3, bucket de GCS, blob de Azure, servidor de archivos HTTP, o incluso un compartido NFS. Registra artefactos fuera de un [W&B Run](https://docs.wandb.ai/ref/python/run) con el [CLI](https://docs.wandb.ai/ref/cli) de W&B.

### Registrar artefactos fuera de runs

W&B crea un run cuando registras un artefacto fuera de un run. Cada artefacto pertenece a un run, que a su vez pertenece a un proyecto; un artefacto (versión) también pertenece a una colección y tiene un tipo.

Usa el comando [`wandb artifact put`](https://docs.wandb.ai/ref/cli/wandb-artifact/wandb-artifact-put) para subir un artefacto al servidor de W&B fuera de un W&B run. Proporciona el nombre del proyecto al que quieres que pertenezca el artefacto junto con el nombre del artefacto (`proyecto/nombre_del_artefacto`). Opcionalmente proporciona el tipo (`TYPE`). Reemplaza `PATH` en el fragmento de código a continuación con la ruta de archivo del artefacto que deseas subir.

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

W&B creará un nuevo proyecto si el proyecto que especificas no existe. Para obtener información sobre cómo descargar un artefacto, consulta [Descargar y usar artefactos](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact).

## Seguimiento de artefactos fuera de W&B

Usa W&B Artifacts para el versionamiento de datasets y el linaje del modelo, y utiliza **artefactos de referencia** para hacer seguimiento de archivos guardados fuera del servidor de W&B. En este modo, un artefacto solo almacena metadatos sobre los archivos, como URLs, tamaño y checksums. Los datos subyacentes nunca abandonan tu sistema. Consulta el [Inicio rápido](https://docs.wandb.ai/guides/artifacts/artifacts-walkthrough) para obtener información sobre cómo guardar archivos y directorios en los servidores de W&B en su lugar.

Lo siguiente describe cómo construir artefactos de referencia y cómo incorporarlos mejor en tus flujos de trabajo.

### Referencias de Amazon S3 / GCS / Azure Blob Storage

Usa W&B Artifacts para el versionamiento de datasets y modelos para hacer seguimiento de referencias en buckets de almacenamiento en la nube. Con las referencias de artefactos, agrega seguimiento de manera transparente sobre tus buckets sin modificaciones en tu estructura de almacenamiento existente.


Los artefactos abstraen el proveedor de almacenamiento en la nube subyacente (tal como AWS, GCP o Azure). La información descrita en la sección siguiente aplica uniformemente para Amazon S3, Google Cloud Storage y Azure Blob Storage.

:::info
Los artefactos de W&B admiten cualquier interfaz compatible con Amazon S3, ¡incluyendo MinIO! Los scripts a continuación funcionan tal como están, cuando configuras la variable de entorno AWS\_S3\_ENDPOINT\_URL para apuntar a tu servidor MinIO.
:::

Supongamos que tenemos un bucket con la siguiente estructura:

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

Bajo `mnist/` tenemos nuestro dataset, una colección de imágenes. Vamos a hacerle seguimiento con un artefacto:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
:::caution
Por defecto, W&B impone un límite de 10,000 objetos al agregar un prefijo de objeto. Puedes ajustar este límite especificando `max_objects=` en llamadas a `add_reference`.
:::

Nuestro nuevo artefacto de referencia `mnist:latest` luce y se comporta de manera similar a un artefacto regular. La única diferencia es que el artefacto solo consta de metadatos sobre el objeto S3/GCS/Azure como su ETag, tamaño y ID de versión (si la versionamiento de objetos está habilitado en el bucket).

W&B usará el mecanismo predeterminado para buscar credenciales basado en el proveedor de nube que uses. Lee la documentación de tu proveedor de nube para aprender más sobre las credenciales utilizadas:

| Proveedor de nube | Documentación de credenciales |
| -------------- | ------------------------- |
| AWS            | [Documentación de Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Documentación de Google Cloud](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Documentación de Azure](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

Interactúa con este artefacto de manera similar a un artefacto normal. En la interfaz de usuario de la aplicación, puedes navegar por el contenido del artefacto de referencia usando el navegador de archivos, explorar el grafo de dependencia completo y revisar el historial versionado de tu artefacto.

:::caution
Medios ricos como imágenes, audio, video y nubes de puntos pueden fallar al renderizarse en la interfaz de usuario de la aplicación dependiendo de la configuración de CORS de tu bucket. Permitir listar **app.wandb.ai** en la configuración de CORS de tu bucket permitirá que la interfaz de usuario de la aplicación renderice correctamente dichos medios ricos.

Los paneles pueden fallar al renderizarse en la interfaz de usuario de la aplicación para buckets privados. Si tu empresa tiene una VPN, podrías actualizar la política de acceso de tu bucket para incluir en la lista blanca las IPs dentro de tu VPN.
:::

### Descargar un artefacto de referencia

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B usará los metadatos registrados cuando se registró el artefacto para recuperar los archivos del bucket subyacente cuando descarga un artefacto de referencia. Si tu bucket tiene habilitado el versionamiento de objetos, W&B recuperará la versión del objeto correspondiente al estado del archivo en el momento en que se registró un artefacto. Esto significa que a medida que evolucionas el contenido de tu bucket, aún puedes apuntar a la iteración exacta de tus datos en la que se entrenó un modelo dado, ya que el artefacto sirve como una instantánea de tu bucket en el momento del entrenamiento.

:::info
W&B recomienda que habilites el 'Versionamiento de Objetos' en tus buckets de almacenamiento si sobrescribes archivos como parte de tu flujo de trabajo. Con el versionamiento habilitado en tus buckets, los artefactos con referencias a archivos que han sido sobrescritos seguirán intactos porque se retienen las versiones antiguas de los objetos.

Basado en tu caso de uso, lee las instrucciones para habilitar el versionamiento de objetos: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable).
:::

### Atándolo todo

El siguiente ejemplo de código demuestra un flujo de trabajo simple que puedes usar para hacer seguimiento de un dataset en Amazon S3, GCS o Azure que alimenta un trabajo de entrenamiento:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# Hacer seguimiento del artefacto y marcarlo como una entrada a
# este run de una sola vez. Una nueva versión del artefacto
# solo se registra si los archivos en el bucket cambiaron.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# Realizar entrenamiento aquí...
```

Para hacer seguimiento de modelos, podemos registrar el artefacto del modelo después de que el script de entrenamiento suba los archivos del modelo al bucket:

```python
import boto3
import wandb

run = wandb.init()

# Entrenamiento aquí...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

:::info
Lee los siguientes reportes para un recorrido end-to-end sobre cómo hacer seguimiento de artefactos por referencia para GCP o Azure:

* [Guía para Hacer Seguimiento de Artefactos por Referencia](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Trabajando con Artefactos de Referencia en Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
:::

### Referencias de Sistema de Archivos

Otro patrón común para un acceso rápido a datasets es exponer un punto de montaje NFS a un sistema de archivos remoto en todas las máquinas que ejecutan trabajos de entrenamiento. Esto puede ser una solución aún más simple que un bucket de almacenamiento en la nube porque, desde la perspectiva del script de entrenamiento, los archivos parecen estar justo en tu sistema de archivos local. Afortunadamente, esa facilidad de uso se extiende al usar Artifacts para hacer seguimiento de referencias a sistemas de archivos, montados o no.

Supongamos que tenemos un sistema de archivos montado en `/mount` con la siguiente estructura:

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

Bajo `mnist/` tenemos nuestro dataset, una colección de imágenes. Vamos a hacerle seguimiento con un artefacto:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

Por defecto, W&B impone un límite de 10,000 archivos al agregar una referencia a un directorio. Puedes ajustar este límite especificando `max_objects=` en llamadas a `add_reference`.

Nota la triple barra inclinada en la URL. El primer componente es el prefijo `file://` que denota el uso de referencias de sistema de archivos. El segundo es la ruta a nuestro dataset, `/mount/datasets/mnist/`.

El artefacto resultante `mnist:latest` luce y actúa justo como un artefacto regular. La única diferencia es que el artefacto solo consta de metadatos sobre los archivos, como sus tamaños y checksums MD5. Los archivos en sí nunca abandonan tu sistema.

Puedes interactuar con este artefacto tal como lo harías con un artefacto normal. En la UI, puedes navegar por el contenido del artefacto de referencia usando el navegador de archivos, explorar el grafo de dependencia completo y revisar el historial versionado de tu artefacto. Sin embargo, la UI no podrá renderizar medios ricos como imágenes, audio, etc., ya que los datos en sí no están contenidos dentro del artefacto.

Descargar un artefacto de referencia es simple:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

Para referencias de sistema de archivos, una operación de `download()` copia los archivos de las rutas referenciadas para construir el directorio del artefacto. En el ejemplo anterior, el contenido de `/mount/datasets/mnist` se copiará en el directorio `artifacts/mnist:v0/`. Si un artefacto contiene una referencia a un archivo que fue sobrescrito, entonces `download()` lanzará un error ya que el artefacto ya no puede reconstruirse.

Poniendo todo junto, aquí hay un flujo de trabajo simple que puedes usar para hacer seguimiento de un dataset bajo un sistema de archivos montado que alimenta un trabajo de entrenamiento:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Hacer seguimiento del artefacto y marcarlo como una entrada a
# este run de una sola vez. Una nueva versión del artefacto
# solo se registra si los archivos bajo el directorio
# cambiaron.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# Realizar entrenamiento aquí...
```

Para hacer seguimiento de modelos, podemos registrar el artefacto del modelo después de que el script de entrenamiento escriba los archivos del modelo en el punto de montaje:

```python
import wandb

run = wandb.init()

# Entrenamiento aquí...

# Escribir modelo en disco

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```