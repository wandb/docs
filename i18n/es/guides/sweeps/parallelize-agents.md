---
description: Parallelize W&B Sweep agents on multi-core or multi-GPU machine.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Paralelizar agentes

<head>
  <title>Paralelizar agentes</title>
</head>

Paraleliza tus agentes de barrido W&B en una máquina con múltiples núcleos o múltiples GPU. Antes de comenzar, asegúrate de haber inicializado tu barrido W&B. Para más información sobre cómo inicializar un barrido W&B, consulta [Inicializar barridos](./initialize-sweeps.md).

### Paralelizar en una máquina con múltiples CPU

Dependiendo de tu caso de uso, explora las pestañas siguientes para aprender cómo paralelizar agentes de barrido W&B usando la CLI o dentro de un Jupyter Notebook.


<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>
  <TabItem value="cli_text">

Usa el comando [`wandb agent`](../../ref/cli/wandb-agent.md) para paralelizar tu agente de barrido W&B a través de múltiples CPU con el terminal. Proporciona el ID de barrido que se devolvió cuando [inicializaste el barrido](./initialize-sweeps.md). 

1. Abre más de una ventana de terminal en tu máquina local.
2. Copia y pega el fragmento de código a continuación y reemplaza `sweep_id` con tu ID de barrido:


```bash
wandb agent sweep_id
```


  </TabItem>
  <TabItem value="jupyter">

Usa la biblioteca SDK de Python de W&B para paralelizar tu agente de barrido W&B a través de múltiples CPU dentro de Jupyter Notebooks. Asegúrate de tener el ID de barrido que se devolvió cuando [inicializaste el barrido](./initialize-sweeps.md). Además, proporciona el nombre de la función que el barrido ejecutará para el parámetro `function`:

1. Abre más de un Jupyter Notebook.
2. Copia y pega el ID de barrido W&B en múltiples Jupyter Notebooks para paralelizar un barrido W&B. Por ejemplo, puedes pegar el siguiente fragmento de código en múltiples cuadernos jupyter para paralelizar tu barrido si tienes el ID de barrido almacenado en una variable llamada `sweep_id` y el nombre de la función es `function_name`: 


```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### Paralelizar en una máquina con múltiples GPU

Sigue el procedimiento descrito para paralelizar tu agente de barrido W&B a través de múltiples GPU con un terminal usando CUDA Toolkit:

1. Abre más de una ventana de terminal en tu máquina local.
2. Especifica la instancia de GPU a usar con `CUDA_VISIBLE_DEVICES` cuando inicies un trabajo de barrido W&B ([`wandb agent`](../../ref/cli/wandb-agent.md)). Asigna a `CUDA_VISIBLE_DEVICES` un valor entero correspondiente a la instancia de GPU a usar.

Por ejemplo, supón que tienes dos GPU NVIDIA en tu máquina local. Abre una ventana de terminal y establece `CUDA_VISIBLE_DEVICES` en `0` (`CUDA_VISIBLE_DEVICES=0`). Reemplaza `sweep_ID` en el ejemplo siguiente con el ID de barrido W&B que se devuelve cuando inicializaste un barrido W&B:

Terminal 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

Abre una segunda ventana de terminal. Establece `CUDA_VISIBLE_DEVICES` en `1` (`CUDA_VISIBLE_DEVICES=1`). Pega el mismo ID de barrido W&B para el `sweep_ID` mencionado en el fragmento de código siguiente:

Terminal 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```