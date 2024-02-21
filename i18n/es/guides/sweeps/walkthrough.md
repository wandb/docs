---
description: Sweeps quickstart shows how to define, initialize, and run a sweep. There
  are four main steps
displayed_sidebar: default
---

# Recorrido

<head>
  <title>Recorrido por los barridos</title>
</head>

Esta página muestra cómo definir, inicializar y ejecutar un barrido. Hay cuatro pasos principales:

1. [Configura tu código de entrenamiento](#set-up-your-training-code)
2. [Define el espacio de búsqueda con una configuración de barrido](#define-the-search-space-with-a-sweep-configuration)
3. [Inicializa el barrido](#initialize-the-sweep)
4. [Inicia el agente de barrido](#start-the-sweep)

Copia y pega el siguiente código en un Jupyter Notebook o script de Python:

```python 
# Importa la Biblioteca de Python de W&B e inicia sesión en W&B
import wandb

wandb.login()

# 1: Define la función objetivo/de entrenamiento
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define el espacio de búsqueda
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: Inicia el barrido
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

Las siguientes secciones desglosan y explican cada paso en el ejemplo de código.

## Configura tu código de entrenamiento
Define una función de entrenamiento que toma valores de hiperparámetros de `wandb.config` y los utiliza para entrenar un modelo y devolver métricas.

Opcionalmente proporciona el nombre del proyecto donde quieres que se almacene el resultado del W&B Run (parámetro del proyecto en [`wandb.init`](../../ref/python/init.md)). Si el proyecto no se especifica, el run se coloca en un proyecto "Sin categorizar".

:::tip
Tanto el barrido como el run deben estar en el mismo proyecto. Por lo tanto, el nombre que proporciones cuando inicializas W&B debe coincidir con el nombre del proyecto que proporcionas cuando inicializas un barrido.
:::

```python
# 1: Define la función objetivo/de entrenamiento
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## Define el espacio de búsqueda con una configuración de barrido
Dentro de un diccionario, especifica sobre qué hiperparámetros quieres hacer el barrido. Para obtener más información sobre las opciones de configuración, consulta [Define la configuración de barrido](./define-sweep-configuration.md).

El ejemplo siguiente demuestra una configuración de barrido que utiliza una búsqueda aleatoria (`'method':'random'`). El barrido seleccionará aleatoriamente un conjunto de valores listados en la configuración para el tamaño del lote, epoch y la tasa de aprendizaje.

A lo largo de los barridos, W&B maximizará la métrica especificada en la clave de métrica (`metric`). En el siguiente ejemplo, W&B maximizará (`'goal':'maximize'`) la precisión de validación (`'val_acc'`).


```python
# 2: Define el espacio de búsqueda
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## Inicializa el barrido

W&B utiliza un _Controlador de Barrido_ para gestionar barridos en la nube (estándar), localmente (local) en una o más máquinas. Para obtener más información sobre los Controladores de Barrido, consulta [Algoritmos de búsqueda y parada localmente](./local-controller.md).

Se devuelve un número de identificación de barrido cuando inicializas un barrido:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

Para obtener más información sobre cómo inicializar barridos, consulta [Inicializa barridos](./initialize-sweeps.md).

## Inicia el barrido

Utiliza la llamada a la API [`wandb.agent`](../../ref/python/agent.md) para iniciar un barrido.

```python
wandb.agent(sweep_id, function=main, count=10)
```

## Visualizar resultados (opcional)

Abre tu proyecto para ver tus resultados en vivo en el panel de control de la App de W&B. Con solo unos pocos clics, construye gráficos interactivos y ricos, como [gráficos de coordenadas paralelas](../app/features/panels/parallel-coordinates.md), [análisis de importancia de parámetros](../app/features/panels/parameter-importance.md), y [más](../app/features/panels/intro.md).

![Ejemplo del panel de control de barridos](/images/sweeps/quickstart_dashboard_example.png)

Para obtener más información sobre cómo visualizar resultados, consulta [Visualiza resultados de barridos](./visualize-sweep-results.md). Para ver un ejemplo de panel de control, consulta este ejemplo de [Proyecto de Barridos](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3).

## Detener el agente (opcional)

Desde la terminal, presiona `Ctrl+c` para detener el run que el agente de barrido está ejecutando actualmente. Para matar al agente, presiona `Ctrl+c` de nuevo después de que el run se detenga.