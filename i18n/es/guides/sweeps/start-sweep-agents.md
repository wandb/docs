---
description: Start or stop a W&B Sweep Agent on one or more machines.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Iniciar agentes de barrido

<head>
  <title>Iniciar o detener un barrido de W&B</title>
</head>

Inicie un barrido de W&B en uno o más agentes en una o más máquinas. Los agentes de barrido de W&B consultan al servidor de W&B que lanzó cuando inicializó un barrido de W&B (`wandb sweep)` para obtener hiperparámetros y usarlos en el entrenamiento del modelo.

Para iniciar un agente de barrido de W&B, proporcione el ID de barrido de W&B que se devolvió cuando inicializó un barrido de W&B. El ID de barrido de W&B tiene el formato:

```bash
entidad/proyecto/sweep_ID
```

Donde:

* entidad: Su nombre de usuario o nombre de equipo en W&B.
* proyecto: El nombre del proyecto donde desea que se almacene el resultado del Run de W&B. Si no se especifica el proyecto, el run se coloca en un proyecto "No categorizado".
* sweep\_ID: El ID único, pseudoaleatorio generado por W&B.

Proporcione el nombre de la función que el barrido de W&B ejecutará si inicia un agente de barrido de W&B dentro de un Jupyter Notebook o script de Python.

Los siguientes fragmentos de código demuestran cómo iniciar un agente con W&B. Asumimos que ya tiene un archivo de configuración y ya ha inicializado un barrido de W&B. Para obtener más información sobre cómo definir un archivo de configuración, consulte [Definir configuración de barrido](./define-sweep-configuration.md).

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Script de Python o Jupyter Notebook', value: 'python'},
  ]}>
  <TabItem value="cli">

Utilice el comando `wandb agent` para iniciar un barrido. Proporcione el ID de barrido que se devolvió cuando inicializó el barrido. Copie y pegue el fragmento de código a continuación y reemplace `sweep_id` con su ID de barrido:

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

Utilice la biblioteca SDK de Python de W&B para iniciar un barrido. Proporcione el ID de barrido que se devolvió cuando inicializó el barrido. Además, proporcione el nombre de la función que el barrido ejecutará.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### Detener agente de W&B

:::caution
Las búsquedas aleatorias y bayesianas se ejecutarán indefinidamente. Debe detener el proceso desde la línea de comandos, dentro de su script de python, o en la [Interfaz de Usuario de Barridos](./visualize-sweep-results.md).
:::

Especifique opcionalmente el número de Runs de W&B que un agente de barrido debe intentar. Los siguientes fragmentos de código demuestran cómo establecer un número máximo de [Runs de W&B](../../ref/python/run.md) con la CLI y dentro de un Jupyter Notebook, script de Python.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Script de Python o Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

Primero, inicialice su barrido. Para obtener más información, consulte [Inicializar barridos](./initialize-sweeps.md).

```
sweep_id = wandb.sweep(sweep_config)
```

A continuación, inicie el trabajo de barrido. Proporcione el ID de barrido generado desde la iniciación del barrido. Pase un valor entero al parámetro count para establecer el número máximo de runs a intentar.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

:::caution
Si inicia un nuevo run después de que el agente de barrido haya terminado, dentro del mismo script o notebook, entonces debería llamar a `wandb.teardown()` antes de iniciar el nuevo run.
:::


  </TabItem>

  <TabItem value="cli">

Primero, inicialice su barrido con el comando [`wandb sweep`](../../ref/cli/wandb-sweep.md). Para obtener más información, consulte [Inicializar barridos](./initialize-sweeps.md).

```
wandb sweep config.yaml
```

Pase un valor entero a la bandera count para establecer el número máximo de runs a intentar.

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>