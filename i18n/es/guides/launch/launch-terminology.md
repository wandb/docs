---
displayed_sidebar: default
---

# Términos y conceptos
Con W&B Launch, puedes encolar [trabajos](#launch-job) en [colas](#launch-queue) para crear runs. Los trabajos son scripts de python instrumentados con W&B. Las colas contienen una lista de trabajos para ejecutar en un [recurso objetivo](#target-resources). Los [agentes](#launch-agent) extraen trabajos de las colas y ejecutan los trabajos en recursos objetivos. W&B hace seguimiento de los trabajos de lanzamiento de manera similar a cómo hace seguimiento de los [runs](../runs/intro.md).

### Trabajo de lanzamiento
Un trabajo de lanzamiento es un tipo específico de [Artefacto W&B](../artifacts/intro.md) que representa una tarea a completar. Por ejemplo, los trabajos de lanzamiento comunes incluyen entrenar un modelo o desencadenar una evaluación del modelo. Las definiciones de los trabajos incluyen:

- Código Python y otros archivos de activos, incluyendo al menos un punto de entrada ejecutable.
- Información sobre la entrada (parámetro de configuración) y salida (métricas registradas).
- Información sobre el entorno. (por ejemplo, `requirements.txt`, `Dockerfile` base).


Hay tres tipos principales de definiciones de trabajos:


| Tipos de trabajos | Definición | Cómo ejecutar este tipo de trabajo | 
| ---------- | --------- | -------------- |
|Trabajos basados en artefactos (o basados en código)| El código y otros activos se guardan como un artefacto W&B.| Para ejecutar trabajos basados en artefactos, el agente de lanzamiento debe estar configurado con un constructor. |
|Trabajos basados en Git| El código y otros activos se clonan de un cierto commit, rama o etiqueta en un repositorio git. | Para ejecutar trabajos basados en Git, el agente de lanzamiento debe estar configurado con un constructor y credenciales del repositorio git. |
|Trabajos basados en imágenes|El código y otros activos están incorporados en una imagen de Docker. | Para ejecutar trabajos basados en imágenes, el agente de lanzamiento podría necesitar estar configurado con credenciales del repositorio de imágenes. | 


:::tip
Aunque los trabajos de lanzamiento pueden realizar actividades no relacionadas con el entrenamiento de modelos, por ejemplo, desplegar un modelo en un servidor de inferencia Triton, todos los trabajos deben llamar a `wandb.init` para completarse con éxito. Esto crea un run para fines de seguimiento en un espacio de trabajo de W&B.
:::

Encuentra los trabajos que has creado en la aplicación W&B bajo la pestaña `Trabajos` de tu espacio de trabajo del proyecto. Desde allí, los trabajos se pueden configurar y enviar a una [cola de lanzamiento](#launch-queue) para ser ejecutados en una variedad de [recursos objetivos](#target-resources).

### Cola de lanzamiento
Las *colas* de lanzamiento son listas ordenadas de trabajos para ejecutar en un recurso objetivo específico. Las colas de lanzamiento son primero en entrar, primero en salir. (FIFO). No hay un límite práctico para el número de colas que puedes tener, pero una buena guía es una cola por recurso objetivo. Los trabajos pueden ser encolados con la UI de la aplicación W&B, la CLI de W&B o el SDK de Python. Luego, uno o más agentes de lanzamiento pueden ser configurados para extraer elementos de la cola y ejecutarlos en el recurso objetivo de la cola.

### Recursos objetivos
El entorno de cómputo en el que una cola de lanzamiento está configurada para ejecutar trabajos se llama *recurso objetivo*.

W&B Launch soporta los siguientes recursos objetivos:

- [Docker](./setup-launch-docker.md)
- [Kubernetes](./setup-launch-kubernetes.md)
- [AWS SageMaker](./setup-launch-sagemaker.md)
- [GCP Vertex](./setup-vertex.md)

Cada recurso objetivo acepta un conjunto diferente de parámetros de configuración llamados *configuraciones de recursos*. Las configuraciones de recursos toman valores predeterminados definidos por cada cola de lanzamiento, pero pueden ser sobrescritos de forma independiente por cada trabajo. Consulta la documentación de cada recurso objetivo para más detalles.

### Agente de lanzamiento
Los agentes de lanzamiento son programas ligeros y persistentes que periódicamente revisan las colas de lanzamiento en busca de trabajos para ejecutar. Cuando un agente de lanzamiento recibe un trabajo, primero construye o extrae la imagen de la definición del trabajo y luego la ejecuta en el recurso objetivo.

Un agente puede sondear múltiples colas, sin embargo, el agente debe estar configurado adecuadamente para soportar todos los recursos objetivos de respaldo para cada cola que está sondeando.

### Entorno del agente de lanzamiento
El entorno del agente es el entorno donde se está ejecutando un agente de lanzamiento, sondeando en busca de trabajos.

:::info
El entorno de ejecución del agente es independiente del recurso objetivo de una cola. En otras palabras, los agentes pueden ser desplegados en cualquier lugar siempre y cuando estén configurados suficientemente para acceder a los recursos objetivos requeridos.
:::