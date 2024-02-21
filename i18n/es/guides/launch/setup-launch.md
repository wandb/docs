---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Configuración de Launch

Esta página describe los pasos de alto nivel necesarios para configurar W&B Launch:

1. **Configurar una cola**: Las colas son FIFO y poseen una configuración de cola. La configuración de una cola controla dónde y cómo se ejecutan los trabajos en un recurso objetivo.
2. **Configurar un agente**: Los agentes se ejecutan en tu máquina/infraestructura y consultan una o más colas para trabajos de lanzamiento. Cuando se extrae un trabajo, el agente asegura que la imagen esté construida y disponible. Luego, el agente envía el trabajo al recurso objetivo.

## Configurar una cola
Las colas de lanzamiento deben configurarse para apuntar a un recurso objetivo específico junto con cualquier configuración adicional específica para ese recurso. Por ejemplo, una cola de lanzamiento que apunta a un cluster de Kubernetes podría incluir variables de entorno o establecer un espacio de nombres personalizado en su configuración de cola de lanzamiento. Cuando creas una cola, especificarás tanto el recurso objetivo que deseas usar como la configuración para ese recurso.

Cuando un agente recibe un trabajo de una cola, también recibe la configuración de la cola. Cuando el agente envía el trabajo al recurso objetivo, incluye la configuración de la cola junto con cualquier sobrescritura del propio trabajo. Por ejemplo, puedes usar una configuración de trabajo para especificar el tipo de instancia de Amazon SageMaker para esa instancia de trabajo solamente. En este caso, es común usar [plantillas de configuración de cola](./setup-queue-advanced.md#configure-queue-template) como la interfaz de usuario final.

### Crear una cola
1. Navega a Launch App en [wandb.ai/launch](https://wandb.ai/launch). 
2. Haz clic en el botón **crear cola** en la parte superior derecha de la pantalla. 

![](/images/launch/create-queue.gif)

3. Desde el menú desplegable **Entidad**, selecciona la entidad a la que pertenecerá la cola. 
  :::tip
  Si eliges una entidad de equipo, todos los miembros del equipo podrán enviar trabajos a esta cola. Si eliges una entidad personal (asociada con un nombre de usuario), W&B creará una cola privada que solo ese usuario puede usar.
  :::
4. Proporciona un nombre para tu cola en el campo **Cola**. 
5. Desde el menú desplegable **Recurso**, selecciona el recurso de cómputo que deseas que los trabajos agregados a esta cola utilicen.
6. Elige si permitir **Priorización** para esta cola. Si se habilita la priorización, un usuario de tu equipo puede definir una prioridad para su trabajo de lanzamiento cuando los encolan. Los trabajos de mayor prioridad se ejecutan antes que los trabajos de menor prioridad.
7. Proporciona una configuración de recurso en formato JSON o YAML en el campo **Configuración**. La estructura y semántica de tu documento de configuración dependerán del tipo de recurso al que apunta la cola. Para más detalles, consulta la página de configuración dedicada para tu recurso objetivo.

## Configurar un agente de lanzamiento
Los agentes de lanzamiento son procesos de larga duración que consultan una o más colas de lanzamiento para trabajos. Los agentes de lanzamiento extraen trabajos en orden de primero en entrar, primero en salir (FIFO) o en orden de prioridad dependiendo de las colas de las que extraen. Cuando un agente extrae un trabajo de una cola, opcionalmente construye una imagen para ese trabajo. Luego, el agente envía el trabajo al recurso objetivo junto con las opciones de configuración especificadas en la configuración de la cola.

<!-- Future: Insert image -->

:::info
Los agentes son altamente flexibles y se pueden configurar para soportar una amplia variedad de casos de uso. La configuración requerida para tu agente dependerá de tu caso de uso específico. Consulta la página dedicada para [Docker](./setup-launch-docker.md), [Amazon SageMaker](./setup-launch-sagemaker.md), [Kubernetes](./setup-launch-kubernetes.md), o [Vertex AI](./setup-vertex.md).
:::

:::tip
W&B recomienda que inicies agentes con una clave API de [cuenta de servicio](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful), en lugar de la clave API de un usuario específico. Hay dos beneficios al usar la clave API de una cuenta de servicio:
1. El agente no depende de un usuario individual.
2. El autor asociado con un run creado a través de Launch es visto por Launch como el usuario que envió el trabajo de lanzamiento, en lugar del usuario asociado con el agente.
:::

### Configuración del agente
Configura el agente de lanzamiento con un archivo YAML llamado `launch-config.yaml`. Por defecto, W&B busca el archivo de configuración en `~/.config/wandb/launch-config.yaml`. Opcionalmente, puedes especificar un directorio diferente cuando activas el agente de lanzamiento.

El contenido del archivo de configuración del agente de lanzamiento dependerá del entorno del agente de lanzamiento, el recurso objetivo de la cola de lanzamiento, los requisitos del constructor de Docker, los requisitos del registro en la nube, y así sucesivamente. 

Independientemente de tu caso de uso, hay opciones configurables básicas para el agente de lanzamiento:
* `max_jobs`: número máximo de trabajos que el agente puede ejecutar en paralelo 
* `entity`: la entidad a la que pertenece la cola
* `queues`: el nombre de una o más colas que el agente supervisará

:::tip
Puedes usar la CLI de W&B para especificar opciones configurables universales para el agente de lanzamiento (en lugar del archivo de configuración YAML): número máximo de trabajos, entidad de W&B y colas de lanzamiento. Consulta el comando [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) para más información.
:::


El siguiente fragmento de YAML muestra cómo especificar las claves de configuración básicas del agente de lanzamiento:

```yaml title="launch-config.yaml"
# Máximo número de runs concurrentes a realizar. -1 = sin límite
max_jobs: -1

entity: <nombre-entidad>

# Lista de colas para consultar.
queues:
  - <nombre-cola>
```

### Configurar un constructor de contenedores
El agente de lanzamiento se puede configurar para construir imágenes. Debes configurar el agente para usar un constructor de contenedores si tienes la intención de usar trabajos de lanzamiento creados a partir de repositorios git o artefactos de código. Consulta [Crear un trabajo de lanzamiento](./create-launch-job.md) para más información sobre cómo crear un trabajo de lanzamiento. 

W&B Launch admite tres opciones de constructor:

* Docker: El constructor Docker utiliza un daemon local de Docker para construir imágenes.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko es un proyecto de Google que permite la construcción de imágenes en entornos donde no está disponible un daemon de Docker.
* Noop: El agente no intentará construir trabajos, y en su lugar solo extraerá imágenes preconstruidas.

:::tip
Usa el constructor Kaniko si tu agente está consultando en un entorno donde no está disponible un daemon de Docker (por ejemplo, un cluster de Kubernetes).

Consulta [Configurar Kubernetes](./setup-launch-kubernetes.md) para detalles sobre el constructor Kaniko.
:::

Para especificar un constructor de imágenes, incluye la clave del constructor en tu configuración del agente. Por ejemplo, el siguiente fragmento de código muestra una parte de la configuración de lanzamiento (`launch-config.yaml`) que especifica usar Docker o Kaniko:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### Configurar un registro de contenedores
En algunos casos, es posible que quieras conectar un agente de lanzamiento a un registro en la nube. Escenarios comunes en los que podrías querer conectar un agente de lanzamiento a un registro en la nube incluyen:

* Quieres ejecutar un trabajo en un entorno diferente al que lo construiste, como una estación de trabajo potente o un cluster.
* Quieres usar el agente para construir imágenes y ejecutar estas imágenes en Amazon SageMaker o VertexAI.
* Quieres que el agente de lanzamiento proporcione credenciales para extraer de un repositorio de imágenes.

Para obtener más información sobre cómo configurar el agente para interactuar con un registro de contenedores, consulta la página de [Configuración avanzada del agente](./setup-agent-advanced.md).

## Activar el agente de lanzamiento
Activa el agente de lanzamiento con el comando `launch-agent` de la CLI de W&B:

```bash
wandb launch-agent -q <cola-1> -q <cola-2> --max-jobs 5
```

En algunos casos de uso, es posible que quieras tener un agente de lanzamiento consultando colas desde dentro de un cluster de Kubernetes. Consulta la [página de configuración avanzada de colas](./setup-queue-advanced.md) para más información.