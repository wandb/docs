---
description: Manage storage, memory allocation of W&B Artifacts.
displayed_sidebar: default
---

# Almacenamiento

<head>
    <title>Almacenamiento de Artefactos</title>
</head>

W&B almacena archivos de artefactos en un bucket privado de Google Cloud Storage ubicado en los Estados Unidos por defecto. Todos los archivos están encriptados en reposo y en tránsito.

Para archivos sensibles, recomendamos configurar [Alojamiento Privado](../hosting/intro.md) o usar [artefactos de referencia](./track-external-files.md).

Durante el entrenamiento, W&B guarda localmente registros, artefactos y archivos de configuración en los siguientes directorios locales:

| Archivo   | Ubicación predeterminada | Para cambiar la ubicación predeterminada establecer:              |
| --------- | ------------------------ | ----------------------------------------------------------------- |
| registros | `./wandb`                | `dir` en `wandb.init` o establecer la variable de entorno `WANDB_DIR` |
| artefactos| `~/.cache/wandb`         | la variable de entorno `WANDB_CACHE_DIR`                          |
| configuraciones | `~/.config/wandb` | la variable de entorno `WANDB_CONFIG_DIR`                       |


:::caution
Dependiendo de la máquina en la que se inicialice `wandb`, estos directorios predeterminados pueden no estar ubicados en una parte del sistema de archivos en la que se pueda escribir. Esto puede desencadenar un error.
:::

### Limpiar la caché local de artefactos

W&B almacena en caché archivos de artefactos para acelerar las descargas entre versiones que comparten archivos en común. Con el tiempo, este directorio de caché puede volverse grande. Ejecute el comando [`wandb artifact cache cleanup`](../../ref/cli/wandb-artifact/wandb-artifact-cache/README.md) para podar la caché y para eliminar cualquier archivo que no se haya utilizado recientemente.

El siguiente fragmento de código demuestra cómo limitar el tamaño de la caché a 1GB. Copie y pegue el fragmento de código en su terminal:

```bash
$ wandb artifact cache cleanup 1GB
```

### ¿Cuánto almacenamiento utiliza cada versión de artefacto?

Solo los archivos que cambian entre dos versiones de artefactos incurren en un costo de almacenamiento.

![la v1 del artefacto "dataset" solo tiene 2/5 imágenes que difieren, por lo que solo utiliza el 40% del espacio.](@site/static/images/artifacts/artifacts-dedupe.PNG)

Por ejemplo, suponga que crea un artefacto de imagen llamado `animals` que contiene dos archivos de imagen cat.png y dog.png:

```
images
|-- cat.png (2MB) # Añadido en `v0`
|-- dog.png (1MB) # Añadido en `v0`
```

A este artefacto se le asignará automáticamente una versión `v0`.

Si agrega una nueva imagen `rat.png` a su artefacto, se crea una nueva versión del artefacto, `v1`, y tendrá los siguientes contenidos:

```
images
|-- cat.png (2MB) # Añadido en `v0`
|-- dog.png (1MB) # Añadido en `v0`
|-- rat.png (3MB) # Añadido en `v1`
```

`v1` rastrea un total de 6MB de archivos, sin embargo, solo ocupa 3MB de espacio porque comparte los 3MB restantes en común con `v0`. Si elimina `v1`, recuperará los 3MB de almacenamiento asociados con `rat.png`. Si elimina `v0`, entonces `v1` heredará los costos de almacenamiento de `cat.png` y `dog.png` llevando su tamaño de almacenamiento a 6MB.