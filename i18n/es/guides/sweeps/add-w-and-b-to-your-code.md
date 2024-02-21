---
description: Add W&B to your Python code script or Jupyter Notebook.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Añade W&B a tu código

<head>
  <title>Añade W&B a tu código Python</title>
</head>

Existen numerosas maneras de añadir el SDK de Python de W&B a tu script o Jupyter Notebook. A continuación, se presenta un ejemplo de "mejores prácticas" de cómo integrar el SDK de Python de W&B en tu propio código.

### Script de entrenamiento original

Supongamos que tienes el siguiente código en una celda de Jupyter Notebook o un script de Python. Definimos una función llamada `main` que imita un bucle de entrenamiento típico; para cada epoch, se calcula la precisión y la pérdida en los conjuntos de datos de entrenamiento y validación. Los valores se generan aleatoriamente para el propósito de este ejemplo.

Definimos un diccionario llamado `config` donde almacenamos los valores de los hiperparámetros (línea 15). Al final de la celda, llamamos a la función `main` para ejecutar el código de entrenamiento simulado.

```python showLineNumbers
# train.py
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


config = {"lr": 0.0001, "bs": 16, "epochs": 5}


def main():
    # Nota que definimos valores de `wandb.config`
    # en lugar de definir valores fijos
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("precisión de entrenamiento:", train_acc, "pérdida de entrenamiento:", train_loss)
        print("precisión de validación:", val_acc, "pérdida de entrenamiento:", val_loss)


# Llama a la función principal.
main()
```

### Script de entrenamiento con el SDK de Python de W&B

Los siguientes ejemplos de código demuestran cómo añadir el SDK de Python de W&B a tu código. Si inicias trabajos de barrido de W&B en la CLI, querrás explorar la pestaña CLI. Si inicias trabajos de barrido de W&B dentro de un Jupyter notebook o script de Python, explora la pestaña SDK de Python.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Script de Python o Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  Para crear un barrido de W&B, añadimos lo siguiente al ejemplo de código:

1. Línea 1: Importa el SDK de Python de Weights & Biases.
2. Línea 6: Crea un objeto diccionario donde los pares clave-valor definen la configuración de barrido. En el ejemplo siguiente, el tamaño de lote (`batch_size`), epochs (`epochs`) y los hiperparámetros de tasa de aprendizaje (`lr`) varían durante cada barrido. Para más información sobre cómo crear una configuración de barrido, consulta [Define la configuración de barrido](./define-sweep-configuration.md).
3. Línea 19: Pasa el diccionario de configuración de barrido a [`wandb.sweep`](../../ref/python/sweep). Esto inicializa el barrido. Esto devuelve un ID de barrido (`sweep_id`). Para más información sobre cómo inicializar barridos, consulta [Inicializar barridos](./initialize-sweeps.md).
4. Línea 33: Utiliza la API [`wandb.init()`](../../ref/python/init.md) para generar un proceso en segundo plano para sincronizar y registrar datos como un [W&B Run](../../ref/python/run.md).
5. Línea 37-39: (Opcional) define valores de `wandb.config` en lugar de definir valores codificados.
6. Línea 45: Registra la métrica que queremos optimizar con [`wandb.log`](../../ref/python/log.md). Debes registrar la métrica definida en tu configuración. Dentro del diccionario de configuración (`sweep_configuration` en este ejemplo) definimos el barrido para maximizar el valor de `val_acc`).
7. Línea 54: Inicia el trabajo de barrido con la llamada a la API [`wandb.agent`](../../ref/python/agent.md). Proporciona el ID de barrido (línea 19), el nombre de la función que el barrido ejecutará (`function=main`), y establece el número máximo de runs a intentar a cuatro (`count=4`). Para más información sobre cómo iniciar W&B Sweep, consulta [Iniciar agentes de barrido](./start-sweep-agents.md).


```python showLineNumbers
import wandb
import numpy as np
import random

# Define la configuración del barrido
sweep_configuration = {
    "method": "random",
    "name": "barrido",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

# Inicializa el barrido pasando la config.
# (Opcional) Proporciona un nombre del proyecto.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="mi-primer-barrido")


# Define la función de entrenamiento que toma
# valores de hiperparámetros de `wandb.config` y los utiliza para entrenar un
# modelo y devolver métrica
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    run = wandb.init()

    # nota que definimos valores de `wandb.config`
    # en lugar de definir valores fijos
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


# Inicia el trabajo de barrido.
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

  Para crear un barrido de W&B, primero creamos un archivo de configuración YAML. El archivo de configuración contiene los hiperparámetros que queremos que el barrido explore. En el ejemplo siguiente, el tamaño de lote (`batch_size`), epochs (`epochs`) y los hiperparámetros de tasa de aprendizaje (`lr`) varían durante cada barrido.
  
```yaml
# config.yaml
program: train.py
method: random
name: barrido
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size: 
    values: [16,32,64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

Para más información sobre cómo crear una configuración de barrido de W&B, consulta [Define la configuración de barrido](./define-sweep-configuration.md).

Nota que debes proporcionar el nombre de tu script de Python para la clave `program` en tu archivo YAML.

A continuación, añadimos lo siguiente al ejemplo de código:

1. Línea 1-2: Importa el SDK de Python de Weights & Biases (`wandb`) y PyYAML (`yaml`). PyYAML se utiliza para leer nuestro archivo de configuración YAML.
2. Línea 18: Lee el archivo de configuración.
3. Línea 21: Utiliza la API [`wandb.init()`](../../ref/python/init.md) para generar un proceso en segundo plano para sincronizar y registrar datos como un [W&B Run](../../ref/python/run.md). Pasamos el objeto de configuración al parámetro config.
4. Línea 25 - 27: Define valores de hiperparámetros de `wandb.config` en lugar de usar valores codificados.
5. Línea 33-39: Registra la métrica que queremos optimizar con [`wandb.log`](../../ref/python/log.md). Debes registrar la métrica definida en tu configuración. Dentro del diccionario de configuración (`sweep_configuration` en este ejemplo) definimos el barrido para maximizar el valor de `val_acc`.


```python showLineNumbers
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # Establece tus hiperparámetros por defecto
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # Nota que definimos valores de `wandb.config`
    # en lugar de  definir valores fijos
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


# Llama a la función principal.
main()
```


Navega a tu CLI. Dentro de tu CLI, establece un número máximo de runs que el agente de barrido debe intentar. Este paso es opcional. En el siguiente ejemplo establecemos el número máximo a cinco.

```bash
NUM=5
```

A continuación, inicializa el barrido con el comando [`wandb sweep`](../../ref/cli/wandb-sweep.md). Proporciona el nombre del archivo YAML. Opcionalmente proporciona el nombre del proyecto para la bandera del proyecto (`--project`):

```bash
wandb sweep --project demo-barrido-cli config.yaml
```

Esto devuelve un ID de barrido. Para más información sobre cómo inicializar barridos, consulta [Inicializar barridos](./initialize-sweeps.md).

Copia el ID de barrido y reemplaza `sweepID` en el fragmento de código siguiente para iniciar el trabajo de barrido con el comando [`wandb agent`](../../ref/cli/wandb-agent.md):

```bash
wandb agent --count $NUM tu-entidad/demo-barrido-cli/sweepID
```

Para más información sobre cómo iniciar trabajos de barrido, consulta [Iniciar trabajos de barrido](./start-sweep-agents.md).
  </TabItem>
</Tabs>

## Consideración al registrar métricas

Asegúrate de registrar explícitamente la métrica que especifiques en tu configuración de barrido en W&B. No registres métricas para tu barrido dentro de un subdirectorio.

Por ejemplo, considera el siguiente pseudocódigo. Un usuario quiere registrar la pérdida de validación (`"val_loss": loss`). Primero pasan los valores a un diccionario (línea 16). Sin embargo, el diccionario pasado a `wandb.log` no accede explícitamente al par clave-valor en el diccionario:

```python title="train.py" showLineNumbers
# Importa la Biblioteca de Python de W&B e inicia sesión en W&B
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entidad>", project="mi-primer-barrido")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss": val_metrics})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="mi-primer-barrido")

wandb.agent(sweep_id, function=main, count=10)
```

En su lugar, accede explícitamente al par clave-valor dentro del diccionario de Python. Por ejemplo, en el código siguiente (línea después de crear un diccionario, especifica el par clave-valor cuando pases el diccionario al método `wandb.log`:

```python title="train.py" showLineNumbers
# Importa la Biblioteca de Python de W&B e inicia sesión en W&B
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entidad>", project="mi-primer-barrido")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss", val_metrics["val_loss"]})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="mi-primer-barrido")

wandb.agent(sweep_id, function=main, count=10)
```