---
description: Hyperparameter search and model optimization with W&B Sweeps
slug: /guides/sweeps
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Ajustar Hiperparámetros

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb"/>

<head>
  <title>Ajustar Hiperparámetros con Barridos</title>
</head>

Utiliza los Barridos de W&B para automatizar la búsqueda de hiperparámetros y visualizar el seguimiento de experimentos rico e interactivo. Elige entre métodos de búsqueda populares como Bayesiano, búsqueda en cuadrícula y aleatoria para explorar el espacio de hiperparámetros. Escala y paraleliza el barrido a través de una o más máquinas.

![Obtén información de grandes experimentos de ajuste de hiperparámetros con paneles de control interactivos.](/images/sweeps/intro_what_it_is.png)

### Cómo funciona
Crea un barrido con dos comandos de [W&B CLI](../../ref/cli/README.md):


1. Inicializa un barrido

```bash
wandb sweep --project <nombre-del-proyecto> <ruta-al-archivo-de-configuración>
```

2. Inicia el agente de barrido

```bash
wandb agent <ID-del-barrido>
```

:::tip
El fragmento de código anterior, y el Colab vinculado en esta página, muestran cómo inicializar y crear un barrido con W&B CLI. Consulta el [Recorrido por los Barridos](./walkthrough.md) para un esquema paso a paso de los comandos de W&B Python SDK a usar para definir una configuración de barrido, inicializar un barrido e iniciar un barrido.
:::

### Cómo empezar

Dependiendo de tu caso de uso, explora los siguientes recursos para comenzar con los Barridos de W&B:

* Si es tu primera vez usando los Barridos de W&B, te recomendamos que pases por el [Colab Notebook de Barridos](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb).
* Lee el [recorrido por los barridos](./walkthrough.md) para un esquema paso a paso de los comandos de W&B Python SDK a usar para definir una configuración de barrido, inicializar un barrido e iniciar un barrido.
* Explora este capítulo para aprender cómo:
  * [Añadir W&B a tu código](./add-w-and-b-to-your-code.md)
  * [Definir la configuración de barrido](./define-sweep-configuration.md)
  * [Inicializar barridos](./initialize-sweeps.md)
  * [Iniciar agentes de barrido](./start-sweep-agents.md)
  * [Visualizar los resultados de barridos](./visualize-sweep-results.md)
* Explora una [lista curada de experimentos de Barridos](./useful-resources.md) que exploran la optimización de hiperparámetros con los Barridos de W&B. Los resultados se almacenan en los Reportes de W&B.

Para un video paso a paso, ve: [Ajustar Hiperparámetros Fácilmente con los Barridos de W&B](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases).