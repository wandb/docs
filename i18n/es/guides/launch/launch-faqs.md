---
description: Answers to frequently asked question about W&B Launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Preguntas frecuentes sobre el lanzamiento

<head>
  <title>Preguntas frecuentes sobre el lanzamiento</title>
</head>

## Empezando

### No quiero que W&B construya un contenedor para mí, ¿puedo seguir usando Launch?
  
Sí. Ejecuta lo siguiente para lanzar una imagen de docker preconstruida. Reemplaza los elementos en `< >` con tu información:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```  

Esto construirá un trabajo cuando crees un run.

O puedes hacer un trabajo a partir de una imagen:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### ¿Existen mejores prácticas para usar Launch de manera efectiva?

  1. Crea tu cola antes de iniciar tu agente, para que puedas configurar tu agente para que apunte fácilmente a ella. Si no haces esto, tu agente dará errores y no funcionará hasta que agregues una cola.
  2. Crea una cuenta de servicio de W&B para iniciar el agente, para que no esté vinculada a una cuenta de usuario individual.
  3. Usa `wandb.config` para leer y escribir tus hiperparámetros, para que puedan ser sobrescritos al volver a ejecutar un trabajo. Consulta [esta guía](https://docs.wandb.ai/guides/launch/create-launch-job#making-your-code-job-friendly) si usas argsparse.

### No me gusta hacer clic, ¿puedo usar Launch sin pasar por la UI?
  
  Sí. El estándar `wandb` CLI incluye un subcomando `launch` que puedes usar para lanzar tus trabajos. Para más información, intenta ejecutar

  ```bash
  wandb launch --help
  ```

### ¿Puede Launch aprovisionar automáticamente (y apagar) recursos de cómputo para mí en el entorno objetivo?

Esto depende del entorno, podemos aprovisionar recursos en SageMaker y Vertex. En Kubernetes, los autoscalers pueden usarse para activar y desactivar recursos automáticamente cuando sea necesario. Los Arquitectos de Soluciones en W&B están felices de trabajar contigo para configurar tu infraestructura subyacente de Kubernetes para facilitar reintentos, autoscaling y uso de pools de nodos de instancias spot. Ponte en contacto con support@wandb.com o tu canal de Slack compartido.

### ¿`wandb launch -d` o `wandb job create image` está subiendo un artefacto docker completo y no lo está extrayendo de un registro? 

No. El comando `wandb launch -d` no subirá a un registro por ti. Necesitas subir tu imagen a un registro tú mismo. Aquí están los pasos generales:

1. Construye una imagen.
2. Empuja la imagen a un registro.

El flujo de trabajo se ve así:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

Desde allí, el agente de lanzamiento iniciará un trabajo apuntando a ese contenedor. Ver [Configuración avanzada del agente](./setup-agent-advanced.md#agent-configuration) para ejemplos de cómo dar al agente acceso para extraer una imagen de un registro de contenedores.

Para Kubernetes, los pods del cluster de Kubernetes necesitarán acceso al registro al que estás empujando.

### ¿Puedo especificar un Dockerfile y dejar que W&B construya una imagen Docker para mí?
Sí. Esto es particularmente útil si tienes muchos requisitos que no cambian a menudo, pero tienes un código base que sí cambia a menudo.

:::important
Asegúrate de que tu Dockerfile esté formateado para usar montajes. Para más información, consulta [la documentación de Montajes en el sitio web de Docker Docs](https://docs.docker.com/build/guide/mounts/). 
:::

Una vez que tu Dockerfile esté configurado, puedes especificar tu Dockerfile de una de tres maneras a W&B:

* Usa Dockerfile.wandb
* CLI de W&B
* Aplicación de W&B


<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'CLI de W&B', value: 'cli'},
    {label: 'Aplicación de W&B', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

Incluye un archivo llamado `Dockerfile.wandb` en el mismo directorio que el punto de entrada del run de W&B. W&B usará `Dockerfile.wandb` en lugar del Dockerfile integrado de W&B.


  </TabItem>
  <TabItem value="cli">

Proporciona la bandera `--dockerfile` cuando llames a la cola un trabajo de lanzamiento con el comando [`wandb launch`](../../ref/cli/wandb-launch.md):

```bash
wandb launch --dockerfile path/to/Dockerfile
```


  </TabItem>
  <TabItem value="app">


Cuando agregues un trabajo a una cola en la Aplicación de W&B, proporciona la ruta a tu Dockerfile en la sección **Overrides**. Más específicamente, proporciónalo como un par clave-valor donde `"dockerfile"` es la clave y el valor es la ruta a tu Dockerfile. 

Por ejemplo, el siguiente JSON muestra cómo incluir un Dockerfile que está dentro de un directorio local:

```json title="Trabajo de lanzamiento en la Aplicación W&B"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>

## Permisos y Recursos

### ¿Cómo controlo quién puede empujar a una cola?

Las colas están limitadas a un equipo de usuarios. Definas la entidad propietaria cuando creas la cola. Para restringir el acceso, puedes cambiar la membresía del equipo.

### ¿Qué permisos requiere el agente en Kubernetes?
“El siguiente manifiesto de kubernetes creará un rol llamado
`wandb-launch-agent` en el`namespace wandb`. Este rol permitirá al agente crear pods, configmaps, secretos y pods/log en el `namespace wandb`. El `wandb-cluster-role` permitirá al agente crear pods, pods/log, secretos, trabajos y trabajos/status en cualquier namespace de tu elección.”

### ¿Launch soporta paralelización? ¿Cómo puedo limitar los recursos consumidos por un trabajo?
   
Sí, Launch soporta escalar trabajos a través de múltiples GPUs y múltiples nodos. Ver [esta guía](https://docs.wandb.ai/tutorials/volcano) para detalles.

En un nivel inter-trabajo, un agente de lanzamiento individual está configurado con un parámetro `max_jobs` que determina cuántos trabajos puede ejecutar ese agente simultáneamente. Además, puedes apuntar a tantos agentes como quieras a una cola particular, siempre que esos agentes estén conectados a una infraestructura en la que puedan lanzarse.
  
Puedes limitar la CPU/GPU, memoria y otros requisitos a nivel de cola de lanzamiento o nivel de ejecución del trabajo, en la configuración de recursos. Para más información sobre la configuración de colas con límites de recursos en Kubernetes ver [aquí](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration). 

   
Para barridos, en el SDK puedes agregar un bloque a la configuración de la cola

```yaml title="configuración de la cola"
  scheduler:
    num_workers: 4
```
Para limitar el número de runs concurrentes de un barrido que se ejecutarán en paralelo.

### Cuando se utilizan colas de Docker para ejecutar múltiples trabajos que descargan el mismo artefacto con `use_artifact`, ¿se vuelve a descargar el artefacto para cada ejecución del trabajo, o hay alguna caché bajo el capó?

No hay caché; cada trabajo es independiente. Sin embargo, hay formas de configurar tu cola/agente donde monta una caché compartida. Puedes lograr esto a través de argumentos de docker en la configuración de la cola.

Como un caso especial, también puedes montar la caché de artefactos de W&B como un volumen persistente.

### ¿Puedes especificar secretos para trabajos/automatizaciones? Por ejemplo, una clave API que no deseas que sea directamente visible para los usuarios?

Sí. La forma sugerida es:

  1. Agrega el secreto como un secreto k8s vainilla en el namespace donde se crearán los runs. algo como `kubectl create secret -n <namespace> generic <secret_name> <secret value>`

 2. Una vez que se crea ese secreto, puedes especificar una configuración de cola para inyectar el secreto cuando comiencen los runs. Los usuarios finales no pueden ver el secreto, solo los administradores del cluster pueden.

### ¿Cómo pueden los administradores restringir lo que los ingenieros de ML tienen acceso a modificar? Por ejemplo, cambiar una etiqueta de imagen puede estar bien, pero otros ajustes de trabajo pueden no estarlo.
  
Esto se puede controlar por [plantillas de configuración de cola](./setup-queue-advanced.md), que exponen ciertos campos de cola para que los usuarios no administradores del equipo editen dentro de los límites definidos por los usuarios administradores. Solo los administradores del equipo pueden crear o editar colas, incluyendo definir qué campos se exponen y los límites para ellos.

### ¿Cómo construye W&B las imágenes?

Los pasos tomados para construir una imagen varían dependiendo de la fuente del trabajo que se está ejecutando, y si la configuración de recursos especifica una imagen base de acelerador.

:::note
Al especificar una configuración de cola, o enviar un trabajo, se puede proporcionar una imagen base de acelerador en la configuración de recursos de la cola o del trabajo:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

Durante el proceso de construcción, se toman las siguientes acciones dependiendo del tipo de trabajo y la imagen base de acelerador proporcionada:

|                                                     | Instalar python usando apt | Instalar paquetes de python | Crear un usuario y workdir | Copiar código en la imagen | Establecer entrypoint |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| Trabajo proveniente de git                          |                          |            X            |             X             |           X          |        X       |
| Trabajo proveniente de código                       |                          |            X            |             X             |           X          |        X       |
| Trabajo proveniente de git y proporcionado imagen de acelerador |             X            |            X            |             X             |           X          |        X       |
| Trabajo proveniente de código y proporcionado imagen de acelerador|             X            |            X            |             X             |           X          |        X       |
| Trabajo proveniente de imagen                        |                          |                         |                           |                      |                |

### ¿Qué requisitos tiene la imagen base de acelerador?
Para trabajos que usan un acelerador, se puede proporcionar una imagen base de acelerador con los componentes de acelerador requeridos instalados. Otros requisitos para la imagen de acelerador proporcionada incluyen:
- Compatibilidad con Debian (el Dockerfile de Launch usa apt-get para buscar python)
- Compatibilidad con el conjunto de instrucciones de hardware de CPU & GPU (Asegúrate de que tu versión de CUDA sea compatible con la GPU que piensas usar)
- Compatibilidad entre la versión de acelerador que proporcionas y los paquetes instalados en tu algoritmo de ML
- Paquetes instalados que requieren pasos adicionales para configurar la compatibilidad con el hardware

### ¿Cómo hago que W&B Launch funcione con Tensorflow en GPU?
Para trabajos que usan tensorflow en GPU, también puede ser necesario especificar una imagen base personalizada para la construcción del contenedor que realizará el agente para que tus runs utilicen correctamente las GPUs. Esto se puede hacer agregando una etiqueta de imagen bajo la clave `builder.accelerator.base_image` a la configuración de recursos. Por ejemplo:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

Nota antes de la versión de wandb: 0.15.6 usa `cuda` en lugar de `accelerator` como la clave padre para `base_image`.

### ¿Puedes usar un repositorio personalizado para paquetes cuando Launch construye la imagen?

Sí. Para hacerlo, añade la siguiente línea a tu `requirements.txt` y reemplaza los valores pasados a `index-url` y `extra-index-url` con tus propios valores:

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```
 
El `requirements.txt` necesita estar definido en la raíz base del trabajo.

## Reencolado automático de runs en preemptions

En algunos casos, puede ser útil configurar trabajos para que se reanuden después de que se interrumpan. Por ejemplo, podrías ejecutar barridos amplios de hiperparámetros en instancias spot, y querer que continúen cuando más instancias spot se activen. Launch puede soportar esta configuración en clusters de Kubernetes.

Si tu cola de Kubernetes está ejecutando un trabajo en un nodo que es pre-emptado por un programador, el trabajo será automáticamente agregado de nuevo al final de la cola para que pueda reanudarse más tarde. Este run reanudado tendrá el mismo nombre que el original, y se puede seguir desde la misma página en la UI que el original. Un trabajo puede ser reencolado automáticamente de esta manera hasta cinco veces.

Launch detecta si un pod es pre-emptado por un programador al verificar si el pod tiene la condición `DisruptionTarget` con una de las siguientes razones:

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

Si el código de tu trabajo está estructurado para permitir la reanudación, permitirá que estos runs reencolados continúen desde donde se dejaron. De lo contrario, los runs comenzarán desde el principio cuando se reencolen. Consulta nuestra guía para [reanudar runs](../runs/resuming.md) para más información.

Actualmente no hay manera de optar por no participar en el reencolado automático de runs para nodos pre-emptados. Sin embargo, si eliminas un run desde la UI o eliminas el nodo directamente, no será reencolado.

El reencolado automático de runs actualmente solo está disponible en colas de Kubernetes; Sagemaker y Vertex aún no son compatibles.