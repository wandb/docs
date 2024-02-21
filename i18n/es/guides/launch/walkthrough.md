---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Recorrido

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Esta guía te mostrará cómo configurar los componentes fundamentales de W&B launch: **trabajos de lanzamiento**, **colas de lanzamiento** y **agentes de lanzamiento**. Al final de este recorrido, habrás logrado:

1. Crear un trabajo de lanzamiento que entrena una red neuronal.
2. Crear una cola de lanzamiento que se utiliza para enviar trabajos para su ejecución en tu máquina local.
3. Crear un agente de lanzamiento que sondea la cola y comienza tu trabajo de lanzamiento con Docker.

:::note
El recorrido descrito en esta página está diseñado para ejecutarse en tu máquina local con Docker.
:::

## Antes de comenzar

Antes de comenzar, asegúrate de haber cumplido con los siguientes requisitos previos:
1. Instalar W&B Python SDK versión 0.14.0 o superior:
    ```bash
    pip install wandb>=0.14.0
    ```
2. Regístrate para obtener una cuenta gratuita en https://wandb.ai/site y luego inicia sesión en tu cuenta de W&B.
3. Instalar Docker. Consulta la [documentación de Docker](https://docs.docker.com/get-docker/) para obtener más información sobre cómo instalar Docker. Asegúrate de que el daemon de docker esté ejecutándose en tu máquina.

## Crear un trabajo de lanzamiento

Los [trabajos de lanzamiento](./launch-terminology#launch-job) son la unidad básica de trabajo en W&B launch. El siguiente código crea un trabajo de lanzamiento desde un [run](../../ref/python/run.md) de W&B utilizando el SDK de Python de W&B.

1. Copia el siguiente código Python en un archivo llamado `train.py`. Guarda el archivo en tu máquina local. Reemplaza `<tu entidad>` con tu entidad de W&B.

    ```python title="train.py"
    import wandb

    config = {"epochs": 10}

    entity = "<tu entidad>"
    project = "launch-quickstart"
    job_name = "ejemplo_recorrido"

    settings = wandb.Settings(job_name=job_name)

    with wandb.init(
        entity=entity, config=config, project=project, settings=settings
    ) as run:
        config = wandb.config
        for epoch in range(1, config.epochs):
            loss = config.epochs / epoch
            accuracy = (1 + (epoch / config.epochs)) / 2
            wandb.log({"loss": loss, "accuracy": accuracy, "epoch": epoch})

        # highlight-next-line
        wandb.run.log_code()
    ```

2. Ejecuta el script de Python y deja que el script se ejecute hasta que se complete:
    ```bash
    python train.py
    ```

Esto creará un trabajo de lanzamiento. En el ejemplo anterior, el trabajo de lanzamiento se creó en un proyecto `launch-quickstart`.

A continuación, agregaremos el nuevo trabajo de lanzamiento creado a una *cola de lanzamiento*.

:::tip
Hay numerosas maneras de crear un trabajo de lanzamiento. Consulta la página [Crear un trabajo de lanzamiento](./create-launch-job.md) para aprender más sobre las diferentes maneras de crear un trabajo de lanzamiento.
:::

## Agrega tu trabajo de lanzamiento a una cola
Una vez que hayas creado un trabajo de lanzamiento, agrega ese trabajo a una [cola de lanzamiento](./launch-terminology.md#launch-queue). Los siguientes pasos describen cómo crear una cola de lanzamiento básica que utilizará un contenedor Docker como su [recurso objetivo](./launch-terminology.md#target-resources):

1. Navega a tu proyecto de W&B.
2. Selecciona la pestaña Jobs en el panel izquierdo (icono de rayo).
3. Pasa el mouse sobre el nombre del trabajo que creaste y selecciona el botón **Launch**.
4. Se deslizará un cajón desde el lado derecho de tu pantalla. Selecciona lo siguiente:
    1. **Versión del trabajo**: la versión del trabajo a lanzar. Dado que solo tenemos una versión, selecciona la versión **@latest** por defecto.
    2. **Sobrescrituras**: nuevos valores para las entradas del trabajo de lanzamiento. Nuestro run tenía un valor en el `wandb.config`: `epochs`. Podemos sobrescribir este valor dentro del campo de sobrescrituras. Para este recorrido, deja el número de epochs tal como está.
    3. **Cola**: la cola en la que lanzar el run. Desde el desplegable, selecciona **Crear una cola 'Starter'**.

![](/images/launch/starter-launch.gif)
5. Una vez que hayas configurado tu trabajo, haz clic en el botón **Launch now** en la parte inferior del cajón para encolar tu trabajo de lanzamiento.


:::tip
Los contenidos de la configuración de tu cola de lanzamiento variarán dependiendo del recurso objetivo de la cola.
:::

## Iniciar un agente de lanzamiento
Para ejecutar un trabajo de lanzamiento, necesitarás un [agente de lanzamiento](./launch-terminology.md#launch-agent) que sondee la cola de lanzamiento a la que se agregó el trabajo. Sigue estos pasos para crear e iniciar un agente de lanzamiento:

1. Desde [wandb.ai/launch](https://wandb.ai/launch) navega a la página de tu cola de lanzamiento.
2. Haz clic en el botón **Add agent**.
3. Aparecerá un modal con un comando de CLI de W&B. Copia este comando y pégalo en tu terminal.

![](/images/launch/activate_starter_queue_agent.png)

En general, el comando para iniciar un agente de lanzamiento es:

```bash
wandb launch-agent -e <nombre-de-la-entidad> -q <nombre-de-la-cola>
```

Dentro de tu terminal, verás al agente comenzar a sondear colas. Espera unos segundos a un minuto y verás a tu agente ejecutar el trabajo de lanzamiento que agregaste.

:::tip
Los agentes de lanzamiento pueden sondear colas en entornos no locales, como un cluster de Kubernetes.
:::

## Ver tu trabajo de lanzamiento

Navega a tu nuevo proyecto **launch-quickstart** en tu cuenta de W&B y abre la pestaña de trabajos desde la navegación en el lado izquierdo de la pantalla.

![](/images/launch/jobs-tab.png)

La página **Jobs** muestra una lista de trabajos de W&B que se crearon a partir de runs ejecutados previamente. Deberías ver un trabajo llamado **job-source-launch-quickstart-train.py:v0**. Haz clic en tu trabajo de lanzamiento para ver las dependencias del código fuente y una lista de runs que fueron creados por el trabajo de lanzamiento.

:::tip
Puedes editar el nombre del trabajo desde la página de trabajos si deseas hacer el trabajo un poco más memorable.
:::