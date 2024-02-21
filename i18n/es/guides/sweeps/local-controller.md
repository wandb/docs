---
description: Search and stop algorithms locally instead of using the W&B cloud-hosted
  service.
displayed_sidebar: default
---

# Buscar y detener algoritmos localmente

<head>
  <title>Buscar y detener algoritmos localmente con agentes de W&B</title>
</head>

El controlador de hiperparámetros es alojado por Weights & Biased como un servicio en la nube por defecto. Los agentes de W&B se comunican con el controlador para determinar el siguiente conjunto de parámetros a usar para el entrenamiento. El controlador también es responsable de ejecutar algoritmos de detención temprana para determinar qué runs pueden ser detenidos.

La característica de controlador local permite al usuario comenzar la búsqueda y detener algoritmos localmente. El controlador local le da al usuario la capacidad de inspeccionar e instrumentar el código para depurar problemas, así como desarrollar nuevas características que pueden ser incorporadas al servicio en la nube.

:::caution
Esta característica se ofrece para apoyar el desarrollo y depuración más rápidos de nuevos algoritmos para la herramienta de barridos. No está destinada para cargas de trabajo reales de optimización de hiperparámetros.
:::

Antes de empezar, debes instalar el SDK de W&B(`wandb`). Escribe el siguiente fragmento de código en tu línea de comandos:

```
pip install wandb sweeps 
```

Los siguientes ejemplos asumen que ya tienes un archivo de configuración y un bucle de entrenamiento definidos en un script de python o Jupyter Notebook. Para más información sobre cómo definir un archivo de configuración, consulta [Definir configuración de barrido](./define-sweep-configuration.md).

### Ejecutar el controlador local desde la línea de comandos

Inicializa un barrido de manera similar a como normalmente lo harías cuando usas controladores de hiperparámetros alojados por W&B como un servicio en la nube. Especifica la bandera de controlador (`controller`) para indicar que quieres usar el controlador local para trabajos de barrido de W&B:

```bash
wandb sweep --controller config.yaml
```

Alternativamente, puedes separar la inicialización de un barrido y la especificación de que quieres usar un controlador local en dos pasos.

Para separar los pasos, primero agrega la siguiente clave-valor a tu archivo de configuración YAML del barrido:

```yaml
controller:
  type: local
```

Después, inicializa el barrido:

```bash
wandb sweep config.yaml
```

Después de haber inicializado el barrido, inicia un controlador con [`wandb controller`](../../ref/python/controller.md):

```bash
# el comando wandb sweep imprimirá un sweep_id
wandb controller {usuario}/{entidad}/{sweep_id}
```

Una vez que hayas especificado que quieres usar un controlador local, inicia uno o más agentes de barrido para ejecutar el barrido. Inicia un barrido de W&B de manera similar a como normalmente lo harías. Consulta [Iniciar agentes de barrido](../../guides/sweeps/start-sweep-agents.md), para más información.

```bash
wandb sweep sweep_ID
```

### Ejecutar un controlador local con el SDK de Python de W&B

Los siguientes fragmentos de código demuestran cómo especificar y usar un controlador local con el SDK de Python de W&B.

La manera más simple de usar un controlador con el SDK de Python es pasar el ID del barrido al método [`wandb.controller`](../../ref/python/controller.md). Después, usa el método `run` de los objetos devueltos para iniciar el trabajo de barrido:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

Si quieres más control del bucle del controlador:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

O incluso más control sobre los parámetros servidos:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

Si quieres especificar tu barrido enteramente con código puedes hacer algo como esto:

```python
import wandb

sweep = wandb.controller()
sweep.configure_search("grid")
sweep.configure_program("train-dummy.py")
sweep.configure_controller(type="local")
sweep.configure_parameter("param1", value=3)
sweep.create()
sweep.run()
```