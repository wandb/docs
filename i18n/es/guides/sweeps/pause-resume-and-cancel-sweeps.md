---
description: Pause, resume, and cancel a W&B Sweep with the CLI.
displayed_sidebar: default
---

# Pausar, reanudar, detener y cancelar barridos

<head>
    <title>Pausar, reanudar, detener o cancelar barridos de W&B</title>
</head>

Pausa, reanuda y cancela un barrido de W&B con la CLI. Pausar un barrido de W&B le indica al agente de W&B que no se deben ejecutar nuevos Runs de W&B hasta que el barrido se reanude. Reanudar un barrido le indica al agente que continúe ejecutando nuevos Runs de W&B. Detener un barrido de W&B le dice al agente de barrido de W&B que deje de crear o ejecutar nuevos Runs de W&B. Cancelar un barrido de W&B le dice al agente de barrido que termine los Runs de W&B que se están ejecutando actualmente y deje de ejecutar nuevos Runs.

En cada caso, proporciona el ID de barrido de W&B que se generó cuando inicializaste un barrido de W&B. Opcionalmente abre una nueva ventana de terminal para ejecutar los comandos siguientes. Una nueva ventana de terminal facilita la ejecución de un comando si un barrido de W&B está imprimiendo declaraciones de salida en tu ventana de terminal actual.

Utiliza la siguiente guía para pausar, reanudar y cancelar barridos.

### Pausar barridos

Pausa un barrido de W&B para que temporalmente deje de ejecutar nuevos Runs de W&B. Usa el comando `wandb sweep --pause` para pausar un barrido de W&B. Proporciona el ID de barrido de W&B que deseas pausar.

```bash
wandb sweep --pause entidad/proyecto/sweep_ID
```

### Reanudar barridos

Reanuda un barrido de W&B pausado con el comando `wandb sweep --resume`. Proporciona el ID de barrido de W&B que deseas reanudar:

```bash
wandb sweep --resume entidad/proyecto/sweep_ID
```

### Detener barridos

Finaliza un barrido de W&B para dejar de ejecutar nuevos Runs de W&B y permitir que los Runs que se están ejecutando terminen.

```bash
wandb sweep --stop entidad/proyecto/sweep_ID
```

### Cancelar barridos

Cancela un barrido para matar todos los runs en ejecución y detener la ejecución de nuevos runs. Usa el comando `wandb sweep --cancel` para cancelar un barrido de W&B. Proporciona el ID de barrido de W&B que deseas cancelar.

```bash
wandb sweep --cancel entidad/proyecto/sweep_ID
```

Para una lista completa de opciones de comandos de CLI, consulta la guía de referencia CLI de [wandb sweep](../../ref/cli/wandb-sweep.md).

### Pausar, reanudar, detener y cancelar un barrido en varios agentes

Pausa, reanuda, detén o cancela un barrido de W&B en varios agentes desde una sola terminal. Por ejemplo, supongamos que tienes una máquina multicore. Después de inicializar un barrido de W&B, abres nuevas ventanas de terminal y copias el ID del barrido en cada nueva terminal.

Dentro de cualquier terminal, utiliza el comando CLI `wandb sweep` para pausar, reanudar, detener o cancelar un barrido de W&B. Por ejemplo, el siguiente fragmento de código demuestra cómo pausar un barrido de W&B en varios agentes con la CLI:

```
wandb sweep --pause entidad/proyecto/sweep_ID
```

Especifica la bandera `--resume` junto con el ID del barrido para reanudar el barrido en tus agentes:

```
wandb sweep --resume entidad/proyecto/sweep_ID
```

Para obtener más información sobre cómo paralelizar agentes de W&B, consulta [Paralelizar agentes](./parallelize-agents.md).