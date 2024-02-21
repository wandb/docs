---
description: Easily scale and manage ML jobs using W&B Launch.
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Lanzamiento

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Escala fácilmente los [runs](../runs/intro.md) de entrenamiento desde tu escritorio a un recurso de cómputo como Amazon SageMaker, Kubernetes y más con W&B Launch. Una vez que W&B Launch está configurado, puedes ejecutar rápidamente scripts de entrenamiento, suites de evaluación de modelos, preparar modelos para inferencia en producción, y más con unos pocos clics y comandos.

## Cómo funciona

Launch está compuesto por tres componentes fundamentales: **trabajos de lanzamiento**, **colas** y **agentes**.

Un [*trabajo de lanzamiento*](./launch-terminology.md#launch-job) es un plano para configurar y ejecutar tareas en tu flujo de trabajo de ML. Una vez que tienes un trabajo de lanzamiento, puedes agregarlo a una [*cola de lanzamiento*](./launch-terminology.md#launch-queue). Una cola de lanzamiento es una cola primero en entrar, primero en salir (FIFO) donde puedes configurar y enviar tus trabajos a un recurso de cómputo objetivo particular, como Amazon SageMaker o un cluster de Kubernetes.

A medida que los trabajos se agregan a la cola, uno o más [*agentes de lanzamiento*](./launch-terminology.md#launch-agent) consultarán esa cola y ejecutarán el trabajo en el sistema objetivo de la cola.

![](/images/launch/launch_overview.png)

Basado en tu caso de uso, tú (o alguien de tu equipo) configurará la cola de lanzamiento de acuerdo a tu [recurso de cómputo objetivo](./launch-terminology.md#target-resources) elegido (por ejemplo, Amazon SageMaker) y desplegará un agente de lanzamiento en tu propia infraestructura.


Consulta la página de [Términos y conceptos](./launch-terminology.md) para más información sobre trabajos de lanzamiento, cómo funcionan las colas, agentes de lanzamiento e información adicional sobre cómo funciona W&B Launch.

## Cómo empezar

Dependiendo de tu caso de uso, explora los siguientes recursos para empezar con W&B Launch:

* Si esta es tu primera vez usando W&B Launch, te recomendamos que sigas la guía [Recorrido](./walkthrough.md).
* Aprende cómo configurar [W&B Launch](./setup-launch.md).
* Crea un [trabajo de lanzamiento](./create-launch-job.md).
* Consulta el [repositorio GitHub de trabajos públicos de W&B Launch](https://github.com/wandb/launch-jobs) para plantillas de tareas comunes como [desplegar en Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluar un LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals), o más.
    * Visualiza trabajos de lanzamiento creados desde este repositorio en este público [`wandb/jobs` proyecto](https://wandb.ai/wandb/jobs/jobs) de proyecto W&B.