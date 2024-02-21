---
description: Troubleshoot common W&B Sweep issues.
displayed_sidebar: default
---

# Solución de problemas con Barridos

<head>
  <title>Solución de problemas con W&B Barridos</title>
</head>

Soluciona mensajes de error comunes con la orientación sugerida.

### `CommError, Run no existe` y `ERROR Error al subir`

Es posible que tu ID de Run de W&B esté definido si se devuelven ambos mensajes de error. Como ejemplo, podrías tener un fragmento de código similar definido en algún lugar de tus Notebooks de Jupyter o script de Python:

```python
wandb.init(id="alguna-cadena")
```

No puedes establecer un ID de Run para los Barridos de W&B porque W&B genera automáticamente IDs únicos y aleatorios para los Runs creados por los Barridos de W&B.

Los IDs de Run de W&B necesitan ser únicos dentro de un proyecto.

Recomendamos que pases un nombre al parámetro name cuando inicialices W&B, si deseas establecer un nombre personalizado que aparecerá en tablas y gráficos. Por ejemplo:

```python
wandb.init(name="un nombre de run legible y útil")
```

### `Cuda fuera de memoria`

Refactoriza tu código para usar ejecuciones basadas en procesos si ves este mensaje de error. Más específicamente, reescribe tu código a un script de Python. Además, llama al agente de barrido de W&B desde la CLI, en lugar del SDK de Python de W&B.

Como ejemplo, supongamos que reescribes tu código a un script de Python llamado `train.py`. Agrega el nombre del script de entrenamiento (`train.py`) a tu archivo de configuración de barrido YAML (`config.yaml` en este ejemplo):

```yaml
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

A continuación, agrega lo siguiente a tu script de Python `train.py`:

```python
if _name_ == "_main_":
    train()
```

Navega a tu CLI e inicializa un Barrido de W&B con wandb sweep:

```shell
wandb sweep config.yaml
```

Toma nota del ID de Barrido de W&B que se devuelve. A continuación, inicia el trabajo de Barrido con [`wandb agent`](../../ref/cli/wandb-agent.md) con la CLI en lugar del SDK de Python ([`wandb.agent`](../../ref/python/agent.md)). Reemplaza `sweep_ID` en el fragmento de código a continuación con el ID de Barrido que se devolvió en el paso anterior:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

El siguiente error generalmente ocurre cuando no registras la métrica que estás optimizando:

```shell
wandb: ERROR Error al llamar a la API de W&B: error anaconda 400: 
{"code": 400, "message": "TypeError: mal tipo de operando para unario -: 'NoneType'"}
```

Dentro de tu archivo YAML o diccionario anidado especifica una clave llamada "metric" para optimizar. Asegúrate de registrar (`wandb.log`) esta métrica. Además, asegúrate de usar el _nombre exacto_ de la métrica que definiste para optimizar dentro de tu script de Python o Notebook de Jupyter. Para más información sobre los archivos de configuración, consulta [Definir configuración de barrido](./define-sweep-configuration.md).