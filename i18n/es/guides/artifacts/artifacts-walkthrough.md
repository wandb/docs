---
description: Artifacts quickstart shows how to create, track, and use a dataset artifact
  with W&B.
displayed_sidebar: default
---

# Recorrido

<head>
  <title>Recorrido</title>
</head>

El siguiente recorrido demuestra los principales comandos del SDK de Python de W&B utilizados para crear, hacer seguimiento y usar un artefacto de dataset desde [W&B Runs](../runs/intro.md).

## 1. Iniciar sesión en W&B

Importa la biblioteca W&B e inicia sesión en W&B. Necesitarás registrarte para obtener una cuenta gratuita de W&B si aún no lo has hecho.

```python
import wandb

wandb.login()
```

## 2. Inicializar un run

Usa la API [`wandb.init()`](../../ref/python/init.md) para generar un proceso en segundo plano para sincronizar y registrar datos como un Run de W&B. Proporciona un nombre de proyecto y un tipo de trabajo:

```python
# Crear un Run de W&B. Aquí especificamos 'dataset' como el tipo de trabajo ya que este ejemplo
# muestra cómo crear un artefacto de dataset.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. Crear un objeto de artefacto

Crea un objeto de artefacto con la API [`wandb.Artifact()`](../../ref/python/artifact.md). Proporciona un nombre para el artefacto y una descripción del tipo de archivo para los parámetros `name` y `type`, respectivamente.

Por ejemplo, el siguiente fragmento de código demuestra cómo crear un artefacto llamado `‘bicycle-dataset’` con una etiqueta `‘dataset’`:

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

Para más información sobre cómo construir un artefacto, consulta [Construir artefactos](./construct-an-artifact.md).

## Agregar el dataset al artefacto

Agrega un archivo al artefacto. Los tipos de archivos comunes incluyen modelos y datasets. El siguiente ejemplo agrega un dataset llamado `dataset.h5` que se guarda localmente en nuestra máquina al artefacto:

```python
# Agregar un archivo al contenido del artefacto
artifact.add_file(local_path="dataset.h5")
```

Reemplaza el nombre del archivo `dataset.h5` en el fragmento de código anterior con la ruta al archivo que quieres agregar al artefacto.

## 4. Registrar el dataset

Usa el método `log_artifact()` de los objetos de run de W&B para tanto guardar tu versión del artefacto como declarar el artefacto como una salida del run.

```python
# Guardar la versión del artefacto en W&B y marcarlo
# como la salida de este run
run.log_artifact(artifact)
```

Un alias `'latest'` se crea por defecto cuando registras un artefacto. Para más información sobre alias de artefactos y versiones, consulta [Crear un alias personalizado](./create-a-custom-alias.md) y [Crear nuevas versiones de artefactos](./create-a-new-artifact-version.md), respectivamente.

## 5. Descargar y usar el artefacto

El siguiente ejemplo de código demuestra los pasos que puedes seguir para usar un artefacto que has registrado y guardado en los servidores de W&B.

1. Primero, inicializa un nuevo objeto de run con **`wandb.init()`.**
2. Segundo, usa el método [`use_artifact()`](../../ref/python/run.md#use_artifact) del objeto de run para decirle a W&B qué artefacto usar. Esto devuelve un objeto de artefacto.
3. Tercero, usa el método [`download()`](../../ref/python/artifact.md#download) del artefacto para descargar el contenido del artefacto.

```python
# Crear un Run de W&B. Aquí especificamos 'entrenamiento' para 'tipo'
# porque usaremos este run para hacer seguimiento del entrenamiento.
run = wandb.init(project="artifacts-example", job_type="training")

# Consultar W&B por un artefacto y marcarlo como entrada a este run
artifact = run.use_artifact("bicycle-dataset:latest")

# Descargar el contenido del artefacto
artifact_dir = artifact.download()
```

Alternativamente, puedes usar la API Pública (`wandb.Api`) para exportar (o actualizar datos) datos ya guardados en W&B fuera de un Run. Consulta [Seguimiento de archivos externos](./track-external-files.md) para más información.