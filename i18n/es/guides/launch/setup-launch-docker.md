---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Configuración para Docker

La siguiente guía describe cómo configurar W&B Launch para usar Docker en una máquina local tanto para el entorno del agente de lanzamiento como para el recurso objetivo de la cola.

Usar Docker para ejecutar trabajos y como entorno del agente de lanzamiento en la misma máquina local es particularmente útil si tu computadora está instalada en una máquina que no tiene un sistema de gestión de cluster (como Kubernetes).

También puedes usar colas de Docker para ejecutar cargas de trabajo en estaciones de trabajo potentes.

:::tip
Esta configuración es común para usuarios que realizan experimentos en su máquina local, o que tienen una máquina remota a la que se conectan por SSH para enviar trabajos de lanzamiento.
:::

Cuando usas Docker con W&B Launch, W&B primero construirá una imagen y luego construirá y ejecutará un contenedor a partir de esa imagen. La imagen se construye con el comando de Docker `docker run <image-uri>`. La configuración de la cola se interpreta como argumentos adicionales que se pasan al comando `docker run`.

## Configurar una cola de Docker

La configuración de la cola de lanzamiento (para un recurso objetivo Docker) acepta las mismas opciones definidas en el comando CLI [`docker run`](../../ref/cli/wandb-docker-run.md).

El agente recibe opciones definidas en la configuración de la cola. El agente luego combina las opciones recibidas con cualquier sobreescritura de la configuración del trabajo de lanzamiento para producir un comando final `docker run` que se ejecuta en el recurso objetivo (en este caso, una máquina local).

Hay dos transformaciones de sintaxis que tienen lugar:

1. Las opciones repetidas se definen en la configuración de la cola como una lista.
2. Las opciones de bandera se definen en la configuración de la cola como un Booleano con el valor `true`.

Por ejemplo, la siguiente configuración de cola:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

Resulta en el siguiente comando `docker run`:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

Los volúmenes se pueden especificar como una lista de cadenas o una única cadena. Usa una lista si especificas varios volúmenes.

Docker pasa automáticamente las variables de entorno, que no tienen asignado un valor, desde el entorno del agente de lanzamiento. Esto significa que, si el agente de lanzamiento tiene una variable de entorno `MY_EXISTING_ENV_VAR`, esa variable de entorno está disponible en el contenedor. Esto es útil si quieres usar otras claves de configuración sin publicarlas en la configuración de la cola.

La bandera `--gpus` del comando `docker run` te permite especificar las GPUs que están disponibles para un contenedor Docker. Para más información sobre cómo usar la bandera `gpus`, consulta la [documentación de Docker](https://docs.docker.com/config/containers/resource_constraints/#gpu).

:::tip
* Instala el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) para usar GPUs dentro de un contenedor Docker.
* Si construyes imágenes a partir de un trabajo de código o fuente de artefacto, puedes sobrescribir la imagen base usada por el [agente](#configure-a-launch-agent-on-a-local-machine) para incluir el NVIDIA Container Toolkit.
  Por ejemplo, dentro de tu cola de lanzamiento, puedes sobrescribir la imagen base a `tensorflow/tensorflow:latest-gpu`:

  ```json
  {
    "builder": {
      "accelerator": {
        "base_image": "tensorflow/tensorflow:latest-gpu"
      }
    }
  }
  ```
:::

## Crear una cola

Crea una cola que use Docker como recurso de cómputo con la CLI de W&B:

1. Navega a la [página de lanzamiento](https://wandb.ai/launch).
2. Haz clic en el botón **Crear Cola**.
3. Selecciona la **Entidad** en la que te gustaría crear la cola.
4. Ingresa un nombre para tu cola en el campo **Nombre**.
5. Selecciona **Docker** como el **Recurso**.
6. Define tu configuración de cola de Docker en el campo **Configuración**.
7. Haz clic en el botón **Crear Cola** para crear la cola.

## Configurar un agente de lanzamiento en una máquina local

Configura el agente de lanzamiento con un archivo de configuración YAML llamado `launch-config.yaml`. Por defecto, W&B buscará el archivo de configuración en `~/.config/wandb/launch-config.yaml`. Opcionalmente, puedes especificar un directorio diferente cuando actives el agente de lanzamiento.

:::tip
Puedes usar la CLI de W&B para especificar opciones configurables básicas para el agente de lanzamiento (en lugar del archivo de configuración YAML): número máximo de trabajos, entidad de W&B y colas de lanzamiento. Consulta el comando [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) para más información.
:::

## Opciones de configuración básicas del agente

Las siguientes pestañas demuestran cómo especificar las opciones de configuración básicas del agente con la CLI de W&B y con un archivo de configuración YAML:

<Tabs
defaultValue="CLI"
values={[
{label: 'CLI de W&B', value: 'CLI'},
{label: 'Archivo de configuración', value: 'config'},
]}>
<TabItem value="CLI">

```bash
wandb launch-agent -q <nombre-cola> --max-jobs <n>
```

  </TabItem>
  <TabItem value="config">

```yaml title="launch-config.yaml"
max_jobs: <n trabajos concurrentes>
queues:
	- <nombre-cola>
```

  </TabItem>
</Tabs>

## Constructores de imágenes Docker

El agente de lanzamiento en tu máquina puede configurarse para construir imágenes Docker. Por defecto, estas imágenes se almacenan en el repositorio de imágenes local de tu máquina. Para permitir que tu agente de lanzamiento construya imágenes Docker, establece la clave `builder` en la configuración del agente de lanzamiento a `docker`:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

Si no quieres que el agente construya imágenes Docker, y en lugar de eso usar imágenes preconstruidas de un registro, establece la clave `builder` en la configuración del agente de lanzamiento a `noop`

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## Registros de contenedores

Launch usa registros de contenedores externos como Dockerhub, Google Container Registry, Azure Container Registry y Amazon ECR.  
Si quieres ejecutar un trabajo en un entorno diferente de donde lo construiste, configura tu agente para poder extraer de un registro de contenedores. 

Para aprender más sobre cómo conectar el agente de lanzamiento con un registro en la nube, consulta la página [Configuración avanzada del agente](./setup-agent-advanced.md#agent-configuration).