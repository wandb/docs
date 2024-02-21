---
description: Learn how to create configuration files for sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Estructura de configuración de barrido

<head>
  <title>Define la configuración de barrido para el ajuste de hiperparámetros.</title>
</head>

Un barrido de W&B combina una estrategia para explorar los valores de hiperparámetros con el código que los evalúa. La estrategia puede ser tan simple como probar todas las opciones o tan compleja como la Optimización Bayesiana y Hyperband ([BOHB](https://arxiv.org/abs/1807.01774)).

Define una configuración de barrido ya sea en un [diccionario de Python](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) o un archivo [YAML](https://yaml.org/). Cómo defines tu configuración de barrido depende de cómo quieras gestionar tu barrido.

:::info
Define tu configuración de barrido en un archivo YAML si quieres inicializar un barrido y empezar un agente de barrido desde la línea de comandos. Define tu barrido en un diccionario de Python si inicializas un barrido y comienzas un barrido completamente dentro de un script de Python o un notebook de Jupyter.
:::

La siguiente guía describe cómo formatear tu configuración de barrido. Consulta [Opciones de configuración de barrido](./sweep-config-keys.md) para una lista completa de las claves de configuración de barrido de nivel superior.

## Estructura básica

Ambas opciones de formato de configuración de barrido (YAML y diccionario de Python) utilizan pares clave-valor y estructuras anidadas.

Usa claves de nivel superior dentro de tu configuración de barrido para definir cualidades de tu búsqueda de barrido, como el nombre del barrido (clave [`name`](./sweep-config-keys.md#name)), los parámetros a buscar (clave [`parameters`](./sweep-config-keys.md#parameters)), la metodología para buscar en el espacio de parámetros (clave [`method`](./sweep-config-keys.md#method)), y más.

Por ejemplo, los siguientes fragmentos de código muestran la misma configuración de barrido definida dentro de un archivo YAML y dentro de un diccionario de Python. Dentro de la configuración de barrido hay cinco claves de nivel superior especificadas: `program`, `name`, `method`, `metric` y `parameters`.

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Script de Python o notebook de Jupyter', value: 'script'},
  ]}>
  <TabItem value="script">

Define un barrido en una estructura de datos de diccionario de Python si defines el algoritmo de entrenamiento en un script de Python o un notebook de Jupyter.

El siguiente fragmento de código almacena una configuración de barrido en una variable llamada `sweep_configuration`:

```python title="train.py"
sweep_configuration = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```
  </TabItem>
  <TabItem value="cli">
Define una configuración de barrido en un archivo YAML si quieres gestionar barridos interactivamente desde la línea de comandos (CLI)

```yaml title="config.yaml"
program: train.py
name: sweepdemo
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [5, 10, 15]
  optimizer:
    values: ["adam", "sgd"]
```
  </TabItem>
</Tabs>

Dentro de la clave `parameters` de nivel superior, las siguientes claves están anidadas: `learning_rate`, `batch_size`, `epoch` y `optimizer`. Para cada una de las claves anidadas que especifiques, puedes proporcionar uno o más valores, una distribución, una probabilidad y más. Para más información, consulta la sección de [parameters](./sweep-config-keys.md#parameters) en [Opciones de configuración de barrido](./sweep-config-keys.md).

## Parámetros doblemente anidados

Las configuraciones de barrido admiten parámetros anidados. Para delinear un parámetro anidado, usa una clave adicional `parameters` bajo el nombre del parámetro de nivel superior. Las configuraciones de barrido admiten anidación de múltiples niveles.

Especifica una distribución de probabilidad para tus variables aleatorias si usas una búsqueda de hiperparámetros bayesiana o aleatoria. Para cada hiperparámetro:

1. Crea una clave `parameters` de nivel superior en tu configuración de barrido.
2. Dentro de la clave `parameters`, anida lo siguiente:
   1. Especifica el nombre del hiperparámetro que quieres optimizar.
   2. Especifica la distribución que quieres usar para la clave `distribution`. Anida el par clave-valor `distribution` debajo del nombre del hiperparámetro.
   3. Especifica uno o más valores para explorar. El valor (o valores) debe estar en línea con la clave de distribución.
      1. (Opcional) Usa una clave adicional `parameters` bajo el nombre del parámetro de nivel superior para delinear un parámetro anidado.

:::caution
Los parámetros anidados definidos en la configuración de barrido sobrescriben las claves especificadas en una configuración de ejecución de W&B.

Por ejemplo, supongamos que inicializas una ejecución de W&B con la siguiente configuración en un script de Python `train.py` (ver Líneas 1-2). A continuación, defines una configuración de barrido en un diccionario llamado `sweep_configuration` (ver Líneas 4-13). Luego pasas el diccionario de configuración de barrido a `wandb.sweep` para inicializar una configuración de barrido (ver Línea 16).


```python title="train.py" showLineNumbers
def main():
    run = wandb.init(config={"nested_param": {"manual_key": 1}})


sweep_configuration = {
    "top_level_param": 0,
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}

# Inicializar barrido pasando la config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# Iniciar trabajo de barrido.
wandb.agent(sweep_id, function=main, count=4)
```
El `nested_param.manual_key` que se pasa cuando se inicializa la ejecución de W&B (línea 2) no es accesible. El `run.config` solo posee los pares clave-valor que están definidos en el diccionario de configuración de barrido (líneas 4-13).
:::

## Plantilla de configuración de barrido


La siguiente plantilla muestra cómo puedes configurar parámetros y especificar restricciones de búsqueda. Reemplaza `hyperparameter_name` con el nombre de tu hiperparámetro y cualquier valor encerrado en `<>`.

```yaml title="config.yaml"
program: <insert>
method: <insert>
parameter:
  hyperparameter_name0:
    value: 0  
  hyperparameter_name1: 
    values: [0, 0, 0]
  hyperparameter_name: 
    distribution: <insert>
    value: <insert>
  hyperparameter_name2:  
    distribution: <insert>
    min: <insert>
    max: <insert>
    q: <insert>
  hyperparameter_name3: 
    distribution: <insert>
    values:
      - <list_of_values>
      - <list_of_values>
      - <list_of_values>
early_terminate:
  type: hyperband
  s: 0
  eta: 0
  max_iter: 0
command:
- ${Command macro}
- ${Command macro}
- ${Command macro}
- ${Command macro}      
```

## Ejemplos de configuración de barrido

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Script de Python o notebook de Jupyter', value: 'notebook'},
  ]}>
  <TabItem value="cli">


```yaml title="config.yaml" 
program: train.py
method: random
metric:
  goal: minimize
  name: loss
parameters:
  batch_size:
    distribution: q_log_uniform_values
    max: 256 
    min: 32
    q: 8
  dropout: 
    values: [0.3, 0.4, 0.5]
  epochs:
    value: 1
  fc_layer_size: 
    values: [128, 256, 512]
  learning_rate:
    distribution: uniform
    max: 0.1
    min: 0
  optimizer:
    values: ["adam", "sgd"]
```

  </TabItem>
  <TabItem value="notebook">

```python title="train.py" 
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
            "min": 32,
            "q": 8,
        },
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "epochs": {"value": 1},
        "fc_layer_size": {"values": [128, 256, 512]},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```

  </TabItem>
</Tabs>

### Ejemplo de hyperband bayesiano
```yaml
program: train.py
method: bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  dropout:
    values: [0.15, 0.2, 0.25, 0.3, 0.4]
  hidden_layer_size:
    values: [96, 128, 148]
  layer_1_size:
    values: [10, 12, 14, 16, 18, 20]
  layer_2_size:
    values: [24, 28, 32, 36, 40, 44]
  learn_rate:
    values: [0.001, 0.01, 0.003]
  decay:
    values: [1e-5, 1e-6, 1e-7]
  momentum:
    values: [0.8, 0.9, 0.95]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
```

Las siguientes pestañas muestran cómo especificar un número mínimo o máximo de iteraciones para `early_terminate`:

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Número mínimo de iteraciones especificado', value: 'min_iter'},
    {label: 'Número máximo de iteraciones especificado', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

Los corchetes para este ejemplo son: `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`, que iguala a `[3, 9, 27, 81]`.
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

Los corchetes para este ejemplo son `[27/eta, 27/eta/eta]`, que iguala a `[9, 3]`.
  </TabItem>
</Tabs>

### Ejemplo de comando
```yaml
program: main.py
metric:
  name: val_loss
  goal: minimize

method: bayes
parameters:
  optimizer.config.learning_rate:
    min: !!float 1e-5
    max: 0.1
  experiment:
    values: [expt001, expt002]
  optimizer:
    values: [sgd, adagrad, adam]

command:
- ${env}
- ${interpreter}
- ${program}
- ${args_no_hyphens}
```

<Tabs
  defaultValue="unix"
  values={[
    {label: 'Unix', value: 'unix'},
    {label: 'Windows', value: 'windows'},
  ]}>
  <TabItem value="unix">

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```
  </TabItem>
  <TabItem value="windows">

```bash
python train.py --param1=value1 --param2=value2
```
  </TabItem>
</Tabs>

Las siguientes pestañas muestran cómo especificar macros de comando comunes:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Establecer intérprete de python', value: 'python'},
    {label: 'Agregar parámetros extra', value: 'parameters'},
    {label: 'Omitir argumentos', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

Elimina la macro `{$interpreter}` y proporciona un valor explícitamente para codificar el intérprete de python. Por ejemplo, el siguiente fragmento de código demuestra cómo hacer esto:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

Lo siguiente muestra cómo agregar argumentos de línea de comandos extra no especificados por parámetros de configuración de barrido:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```

  </TabItem>
  <TabItem value="omit">

Si tu programa no utiliza análisis de argumentos, puedes evitar pasar argumentos completamente y aprovechar que `wandb.init` recoja automáticamente los parámetros de barrido en `wandb.config`:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

Puedes cambiar el comando para pasar argumentos de la manera que herramientas como [Hydra](https://hydra.cc) esperan. Consulta [Hydra con W&B](../integrations/other/hydra.md) para más información.

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>