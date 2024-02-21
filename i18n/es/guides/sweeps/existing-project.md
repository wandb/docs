---
description: Tutorial on how to create sweep jobs from a pre-existing W&B project.
displayed_sidebar: default
---

# Tutorial - Crear barridos desde proyectos existentes

<head>
    <title>Tutorial para crear barridos desde proyectos existentes</title>
</head>

El siguiente tutorial te guiará a través de los pasos sobre cómo crear trabajos de barrido a partir de un proyecto de W&B preexistente. Usaremos el [dataset de Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) para entrenar una red neuronal convolucional de PyTorch en cómo clasificar imágenes. El código requerido y el dataset se encuentran en el repositorio de W&B: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

Explora los resultados en este [Panel de W&B](https://app.wandb.ai/carey/pytorch-cnn-fashion).

## 1. Crear un proyecto

Primero, crea una referencia. Descarga el ejemplo de modelo del dataset de MNIST de PyTorch del repositorio de GitHub de ejemplos de W&B. Luego, entrena el modelo. El script de entrenamiento está dentro del directorio `examples/pytorch/pytorch-cnn-fashion`.

1. Clona este repositorio `git clone https://github.com/wandb/examples.git`
2. Abre este ejemplo `cd examples/pytorch/pytorch-cnn-fashion`
3. Ejecuta un run manualmente `python train.py`

Opcionalmente explora el ejemplo en el dashboard del UI de la App de W&B.

[Ver una página de ejemplo del proyecto →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Crear un barrido

Desde tu [página del proyecto](../app/pages/project-page.md), abre la [pestaña de Barridos](./sweeps-ui.md) en la barra lateral y selecciona **Crear Barrido**.

![](@site/static/images/sweeps/sweep1.png)

La configuración auto-generada adivina los valores sobre los que barrer basándose en los runs que has completado. Edita la configuración para especificar qué rangos de hiperparámetros quieres probar. Cuando lanzas el barrido, se inicia un nuevo proceso en el servidor de barrido alojado de W&B. Este servicio centralizado coordina los agentes, las máquinas que están ejecutando los trabajos de entrenamiento.

![](@site/static/images/sweeps/sweep2.png)

## 3. Lanzar agentes

A continuación, lanza un agente localmente. Puedes lanzar hasta 20 agentes en diferentes máquinas en paralelo si quieres distribuir el trabajo y terminar el trabajo de barrido más rápidamente. El agente imprimirá el conjunto de parámetros que intentará a continuación.

![](@site/static/images/sweeps/sweep3.png)

Ahora estás ejecutando un barrido. La siguiente imagen demuestra cómo se ve el panel de control mientras el trabajo de barrido de ejemplo se está ejecutando. [Ver una página de ejemplo del proyecto →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s\_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090\_1579066494222\_image.png)

## Sembrar un nuevo barrido con runs existentes

Lanza un nuevo barrido utilizando runs que hayas registrado previamente.

1. Abre la tabla de tu proyecto.
2. Selecciona los runs que quieres usar con las casillas de verificación en el lado izquierdo de la tabla.
3. Haz clic en el desplegable para crear un nuevo barrido.

Tu barrido ahora estará configurado en nuestro servidor. Todo lo que necesitas hacer es lanzar uno o más agentes para empezar a ejecutar runs.

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
Si inicias el nuevo barrido como un barrido bayesiano, los runs seleccionados también sembrarán el Proceso Gaussiano.
:::