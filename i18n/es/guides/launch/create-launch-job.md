---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Crear un trabajo de lanzamiento
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Un trabajo es un plano que contiene información contextual sobre un run de W&B del cual se creó; como el código fuente del run, dependencias de software, hiperparámetros, versión del artefacto, y así sucesivamente.

Una vez que tienes un trabajo de lanzamiento, puedes agregarlos a una [cola de lanzamiento](./launch-terminology.md#launch-queue) preconfigurada. El agente de lanzamiento que fue desplegado por ti o alguien de tu equipo, sondea esa cola y envía el trabajo (como una imagen Docker) al recurso de cómputo que fue configurado en la cola de lanzamiento.

Hay tres maneras de crear un trabajo de lanzamiento:

- [Con un script de Python](#create-a-job-with-a-wb-artifact)
- [Con una imagen Docker](#create-a-job-with-a-docker-image)
- [Con un repositorio Git](#create-a-job-with-git)

Las siguientes secciones muestran cómo crear un trabajo basado en cada caso de uso.

## Antes de comenzar

Antes de crear un trabajo de lanzamiento, averigua el nombre de tu cola y la entidad a la que pertenece. Luego, sigue estas instrucciones para averiguar el estado de tu cola y para verificar si un agente está sondeando esa cola:

1. Navega a [wandb.ai/launch](https://wandb.ai/launch).
2. Desde el desplegable **Todas las entidades**, selecciona la entidad a la que pertenece la cola de lanzamiento.
3. De los resultados filtrados, verifica que la cola exista.
4. Pasa el ratón a la derecha de la cola de lanzamiento y selecciona `Ver cola`.
5. Selecciona la pestaña **Agentes**. Dentro de la pestaña **Agentes** verás una lista de ID de Agentes y sus estados. Asegúrate de que uno de los ID de agente tenga un estado de **sondeo**.

## Crear un trabajo con un artefacto de W&B

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'SDK de Python', value: 'sdk'}
]}>
<TabItem value="cli">

Crea un trabajo de lanzamiento con la CLI de W&B.

Asegúrate de que la ruta con tu script de Python tenga un archivo `requirements.txt` con las dependencias de Python requeridas para ejecutar tu código. También se requiere un entorno de ejecución de Python. El entorno de ejecución de Python puede especificarse manualmente con el parámetro de entorno de ejecución o puede detectarse automáticamente de un archivo `runtime.txt` o `.python-version`.

Copia y pega el siguiente fragmento de código. Reemplaza los valores dentro de `"<>"` basado en tu caso de uso:

```bash
wandb job create --project "<nombre-del-proyecto>" -e "<tu-entidad>" \
--name "<nombre-para-el-trabajo>" code "<ruta-a-script/code.py>"
```

Para una lista completa de banderas que puedes usar, consulta la documentación del comando [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md).

:::note
No necesitas usar la función [`run.log_code()`](../../ref/python/run.md#log_code) dentro de tu script de Python cuando creas un trabajo de lanzamiento con la CLI de W&B.
:::

  </TabItem>
  <TabItem value="sdk">

Registra tu código como un artefacto para crear un trabajo de lanzamiento. Para hacerlo, registra tu código en tu run como un artefacto con [`run.log_code()`](../../ref/python/run.md#log_code).

El siguiente código Python de muestra muestra cómo integrar la función `run.log_code()` (ver parte resaltada) en un script de Python.

```python title="create_simple_job.py"
import random
import wandb


def run_training_run(epochs, lr):
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="launch_demo",
        job_type="eval",
        settings=settings,
        entity="<tu-entidad>",
        # Simulando el seguimiento de hiperparámetros
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulando un entrenamiento
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})

    # resaltar-siguiente-línea
    run.log_code()
    run.finish()


run_training_run(epochs=10, lr=0.01)
```

Puedes especificar un nombre para tu trabajo con la variable de entorno `WANDB_JOB_NAME`. También puedes especificar un nombre estableciendo el parámetro `job_name` en `wandb.Settings` y pasándolo a `wandb.init`. Por ejemplo:

```python
settings = wandb.Settings(job_name="mi-nombre-de-trabajo")
wandb.init(settings=settings)
```

Si no especificas un nombre, W&B genera automáticamente un nombre de trabajo de lanzamiento para ti. El nombre del trabajo se formatea de la siguiente manera: `job-<nombre-del-artefacto-de-código>`.

Para más información sobre el comando [`run.log_code()`](../../ref/python/run.md#log_code), consulta la [guía de referencia de la API](../../ref/README.md).

  </TabItem>
</Tabs>

## Crear un trabajo con una imagen Docker

Crea un trabajo con una imagen Docker con la CLI de W&B o creando un contenedor Docker a partir de la imagen. Para crear un trabajo basado en imagen, primero debes crear la imagen Docker. La imagen Docker debe contener el código fuente (como el Dockerfile, archivo requirements.txt, y así sucesivamente) requerido para ejecutar el run de W&B.

Como ejemplo, supón que tienes un directorio llamado [`fashion_mnist_train`](https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train) con la siguiente estructura de directorio:

```
fashion_mnist_train
│   data_loader.py
│   Dockerfile
│   job.py
│   requirements.txt
└───configs
│   │   example.yml
```

Puedes crear una imagen Docker llamada `fashion-mnist` con el comando `docker build`:

```bash
docker build . -t fashion-mnist
```

Para más información sobre cómo construir imágenes Docker, consulta la [documentación de referencia de Docker build](https://docs.docker.com/engine/reference/commandline/build/).

<Tabs
defaultValue="cli"
values={[
{label: 'CLI de W&B', value: 'cli'},
{label: 'Docker run', value: 'build'},
]}>
<TabItem value="cli">

Crea un trabajo de lanzamiento con la CLI de W&B. Copia el siguiente fragmento de código y reemplaza los valores dentro de `"<>"` basado en tu caso de uso:

```bash
wandb job create --project "<nombre-del-proyecto>" --entity "<tu-entidad>" \
--name "<nombre-para-el-trabajo>" image nombre-imagen:etiqueta
```

Para una lista completa de banderas que puedes usar, consulta la documentación del comando [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md).

  </TabItem>
  <TabItem value="build">

Asocia tu run con una imagen Docker. W&B busca una etiqueta de imagen en la variable de entorno `WANDB_DOCKER`, y si `WANDB_DOCKER` está establecida, se crea un trabajo de lanzamiento a partir de la etiqueta de imagen especificada. Asegúrate de que la variable de entorno `WANDB_DOCKER` esté establecida con la etiqueta de imagen completa.

Crea un trabajo de lanzamiento construyendo un contenedor Docker a partir de una imagen Docker. Copia el siguiente fragmento de código y reemplaza los valores dentro de `"<>"` basado en tu caso de uso:

```bash
docker run -e WANDB_PROJECT="<nombre-del-proyecto>" \
-e WANDB_ENTITY="<tu-entidad>" \
-e WANDB_API_KEY="<tu-clave-de-api-de-w&B>" \
-e WANDB_DOCKER="<nombre-de-imagen-docker>" imagen:etiqueta
```

Puedes especificar un nombre para tu trabajo con la variable de entorno `WANDB_JOB_NAME`. W&B genera automáticamente un nombre de trabajo de lanzamiento para ti si no especificas un nombre. W&B asigna un nombre de trabajo con el siguiente formato: `job-<imagen>-<nombre>`.

:::tip
Asegúrate de que esté establecido con la etiqueta de imagen completa. Por ejemplo, si tu agente ejecuta imágenes de un repositorio ECR, deberías establecer `WANDB_DOCKER` con la etiqueta de imagen completa, incluyendo la URL del repositorio ECR: `123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-imagen:desarrollo`. La etiqueta docker, en este caso `'desarrollo'`, se agrega como un alias al trabajo resultante.
:::

  </TabItem>
</Tabs>

## Crear un trabajo con Git

Crea un trabajo basado en Git con W&B Launch. El código y otros activos se clonan de un cierto commit, rama o etiqueta en un repositorio git.

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Autogenerar desde commit de git', value: 'git'},
]}>
<TabItem value="cli">

```bash
wandb job create --project "<nombre-del-proyecto>" --entity "<tu-entidad>" \ 
--name "<nombre-para-el-trabajo>" git https://github.com/nombre-org/repo-name.git \ 
--entry-point "<ruta-a-script/code.py>"
```

Para construir desde una rama o hash de commit, añade el argumento `-g`.

  </TabItem>
  <TabItem value="git">

Asegúrate de que la ruta con tu script de Python tenga un archivo `requirements.txt` con las dependencias de Python requeridas para ejecutar tu código. También se requiere un entorno de ejecución de Python. El entorno de ejecución de Python puede especificarse manualmente con el parámetro de entorno de ejecución o puede detectarse automáticamente de un archivo `runtime.txt` o `.python-version`.

Puedes especificar un nombre para tu trabajo con la variable de entorno `WANDB_JOB_NAME`. Si no especificas un nombre, W&B genera automáticamente un nombre de trabajo de lanzamiento para ti. En este caso, W&B asigna un nombre de trabajo con el siguiente formato: `job-<url-remota-git>-<ruta-a-script>`.

</TabItem>
</Tabs>

### Manejo de URL remotas de Git

La URL remota asociada con un trabajo de lanzamiento puede ser una URL HTTPS o SSH. Las URL remotas de Git típicamente usan los siguientes formatos:

- `https://github.com/organizacion/repositorio.git` (HTTPS)
- `git@github.com:organizacion/repositorio.git` (SSH)

El formato exacto varía según el proveedor de hosting de git.

El formato de URL remota es importante porque determina cómo se accede y autentica el remoto de git. La siguiente tabla describe los requisitos que debes cumplir para acceder y autenticar:

| URL Remota | Requisitos para acceso y autenticación |
| ---------- | -------------------------------------- |
| URL HTTPS  | nombre de usuario y contraseña para autenticarse con el remoto de git |
| URL SSH    | Clave SSH para autenticarse con el remoto de git |


La URL remota de Git se infiere automáticamente del repositorio git local si tu trabajo de lanzamiento se crea automáticamente por un run de W&B. 

Si creas un trabajo manualmente, eres responsable de proporcionar una URL para tu protocolo de transferencia deseado.

## Nombres de trabajos de lanzamiento

Por defecto, W&B genera automáticamente un nombre de trabajo para ti. El nombre se genera dependiendo de cómo se crea el trabajo (GitHub, artefacto de código o imagen Docker). Alternativamente, puedes definir el nombre de un trabajo de lanzamiento con variables de entorno o con el SDK de Python de W&B.

### Nombres predeterminados de trabajos de lanzamiento

La siguiente tabla describe la convención de nombres utilizada por defecto basada en la fuente del trabajo:

| Fuente        | Convención de nombres                   |
| ------------- | --------------------------------------- |
| GitHub        | `job-<url-remota-git>-<ruta-a-script>`  |
| Artefacto de código | `job-<nombre-del-artefacto-de-código>` |
| Imagen Docker  | `job-<nombre-de-la-imagen>`             |

### Nombra tu trabajo de lanzamiento

Nombra tu trabajo con una variable de entorno de W&B o con el SDK de Python de W&B

<Tabs
defaultValue="env_var"
values={[
{label: 'Variable de entorno', value: 'env_var'},
{label: 'SDK de Python de W&B', value: 'python_sdk'},
]}>
<TabItem value="env_var">

Establece la variable de entorno `WANDB_JOB_NAME` con el nombre de trabajo que prefieras. Por ejemplo:

```bash
WANDB_JOB_NAME=nombre-de-trabajo-asombroso
```

  </TabItem>
  <TabItem value="python_sdk">

Define el nombre de tu trabajo con `wandb.Settings`. Luego pasa este objeto cuando inicializas W&B con `wandb.init`. Por ejemplo:

```python
settings = wandb.Settings(job_name="mi-nombre-de-trabajo")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
Para trabajos de imagen docker, el alias de versión se agrega automáticamente como un alias al trabajo.
:::