---
description: Initialize a W&B Sweep
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Inicializar barridos

<head>
  <title>Iniciar un Barrido W&B</title>
</head>

W&B utiliza un _Controlador de Barrido_ para gestionar barridos en la nube (estándar), localmente (local) en una o más máquinas. Después de que un run se completa, el controlador de barrido emitirá un nuevo conjunto de instrucciones describiendo un nuevo run a ejecutar. Estas instrucciones son recogidas por _agentes_ que realmente realizan los runs. En un Barrido W&B típico, el controlador vive en el servidor de W&B. Los agentes viven en _tu(s)_ máquina(s).

Los siguientes fragmentos de código demuestran cómo inicializar barridos con la CLI y dentro de un Jupyter Notebook o script de Python.

:::caution
1. Antes de inicializar un barrido, asegúrate de tener una configuración de barrido definida, ya sea en un archivo YAML o un objeto de diccionario de Python anidado en tu script. Para más información, consulta [Definir configuración de barrido](../../guides/sweeps/define-sweep-configuration.md).
2. Tanto el Barrido W&B como el Run W&B deben estar en el mismo proyecto. Por lo tanto, el nombre que proporciones al inicializar W&B ([`wandb.init`](../../ref/python/init.md)) debe coincidir con el nombre del proyecto que proporcionas al inicializar un Barrido W&B ([`wandb.sweep`](../../ref/python/sweep.md)).
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Script de Python o Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

Usa el SDK de W&B para inicializar un barrido. Pasa el diccionario de configuración de barrido al parámetro `sweep`. Opcionalmente, proporciona el nombre del proyecto para el parámetro de proyecto (`project`) donde quieres que se almacenen los resultados del Run W&B. Si el proyecto no se especifica, el run se coloca en un proyecto "Sin categorizar".

```python
import wandb

# Ejemplo de configuración de barrido
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="nombre-del-proyecto")
```

La función [`wandb.sweep`](../../ref/python/sweep) devuelve el ID del barrido. El ID del barrido incluye el nombre de la entidad y el nombre del proyecto. Toma nota del ID del barrido.
  </TabItem>
  <TabItem value="cli">

Usa la CLI de W&B para inicializar un barrido. Proporciona el nombre de tu archivo de configuración. Opcionalmente, proporciona el nombre del proyecto para la bandera `project`. Si el proyecto no se especifica, el Run W&B se coloca en un proyecto "Sin categorizar".

Usa el comando [`wandb sweep`](../../ref/cli/wandb-sweep) para inicializar un barrido. El siguiente ejemplo de código inicializa un barrido para un proyecto `sweeps_demo` y utiliza un archivo `config.yaml` para la configuración.

```bash
wandb sweep --project sweeps_demo config.yaml
```

Este comando imprimirá un ID de barrido. El ID del barrido incluye el nombre de la entidad y el nombre del proyecto. Toma nota del ID del barrido.
  </TabItem>
</Tabs>