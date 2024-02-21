---
description: Discover how to automate hyperparamter sweeps on launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Barridos en el Lanzamiento

<CTAButtons colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7"/>

Crea un trabajo de ajuste de hiperparámetros ([barridos](../sweeps/intro.md)) con W&B Launch. Con los barridos en lanzamiento, un programador de barridos se envía a una Cola de Lanzamiento con los hiperparámetros especificados para barrer. El programador de barridos comienza tan pronto como es recogido por el agente, lanzando ejecuciones de barridos en la misma cola con hiperparámetros elegidos. Esto continúa hasta que el barrido termina o se detiene.

Puedes usar el motor de programación de Barridos W&B predeterminado o implementar tu propio programador personalizado:

1. Programador de barridos estándar: Utiliza el motor de programación de Barridos W&B predeterminado que controla los [Barridos W&B](../sweeps/intro.md). Los métodos `bayes`, `grid` y `random` conocidos están disponibles.
2. Programador de barridos personalizado: Configura el programador de barridos para que se ejecute como un trabajo. Esta opción permite una personalización completa. Un ejemplo de cómo extender el programador de barridos estándar para incluir más registros se puede encontrar en la sección a continuación.
 
:::note
Esta guía asume que W&B Launch ha sido previamente configurado. Si W&B Launch no está configurado, consulta la sección [cómo comenzar](./intro.md#how-to-get-started) de la documentación de lanzamiento.
:::

:::tip
Recomendamos que crees un barrido en lanzamiento utilizando el método 'básico' si es la primera vez que usas barridos en lanzamiento. Utiliza un programador de barridos en lanzamiento personalizado cuando el motor de programación estándar W&B no cumpla con tus necesidades.
:::

## Crear un barrido con un programador estándar de W&B
Crea Barridos W&B con Launch. Puedes crear un barrido interactivamente con la Aplicación W&B o programáticamente con la CLI de W&B. Para configuraciones avanzadas de barridos de lanzamiento, incluyendo la capacidad de personalizar el programador, usa la CLI. 

:::info
Antes de crear un barrido con W&B Launch, asegúrate de primero crear un trabajo para barrer. Consulta la página [Crear un Trabajo](./create-launch-job.md) para más información. 
:::


<Tabs
  defaultValue="app"
  values={[
    {label: 'Aplicación W&B', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
Crea un barrido interactivamente con la Aplicación W&B.

1. Navega a tu proyecto W&B en la Aplicación W&B.  
2. Selecciona el icono de barridos en el panel izquierdo (imagen de escoba). 
3. A continuación, selecciona el botón **Crear Barrido**.
4. Haz clic en el botón **Configurar Lanzamiento 🚀**.
5. Desde el menú desplegable **Trabajo**, selecciona el nombre de tu trabajo y la versión del trabajo que deseas crear un barrido.
6. Selecciona una cola para ejecutar el barrido usando el menú desplegable **Cola**.
8. Usa el menú desplegable **Prioridad del Trabajo** para especificar la prioridad de tu trabajo de lanzamiento. La prioridad de un trabajo de lanzamiento se establece en "Media" si la cola de lanzamiento no admite priorización.
8. (Opcional) Configura argumentos de sobreescritura para la ejecución o el programador de barridos. Por ejemplo, usando las sobreescrituras del programador, configura el número de ejecuciones concurrentes que el programador gestiona usando `num_workers`.
9. (Opcional) Selecciona un proyecto para guardar el barrido usando el menú desplegable **Proyecto Destino**.
10. Haz clic en **Guardar**
11. Selecciona **Lanzar Barrido**.

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">

Crea programáticamente un Barrido W&B con Launch con la CLI de W&B.

1. Crea una configuración de Barrido
2. Especifica el nombre completo del trabajo dentro de tu configuración de barrido
3. Inicializa un agente de barrido.

:::info
Los pasos 1 y 3 son los mismos pasos que normalmente tomas cuando creas un Barrido W&B.
:::

Por ejemplo, en el siguiente fragmento de código, especificamos `'wandb/jobs/Hello World 2:latest'` para el valor del trabajo:

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: ejemplos de barrido usando trabajos de lanzamiento

method: bayes
metric:
  goal: minimize
  name: loss_metric
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform

# Parámetros opcionales del programador:

# scheduler:
#   num_workers: 1  # ejecuciones de barrido concurrentes
#   docker_image: <imagen base para el programador>
#   resource: <ej. local-container...>
#   resource_args:  # argumentos del recurso pasados a las ejecuciones
#     env: 
#         - WANDB_API_KEY

# Parámetros opcionales de lanzamiento
# launch: 
#    registry: <registro para la extracción de la imagen>
```

Para información sobre cómo crear una configuración de barrido, consulta la página [Definir configuración de barrido](../sweeps/define-sweep-configuration.md).

4. A continuación, inicializa un barrido. Proporciona la ruta a tu archivo de configuración, el nombre de tu cola de trabajos, tu entidad W&B y el nombre del proyecto.

```bash
wandb launch-sweep <ruta/a/archivo/yaml> --queue <nombre_cola> --entity <tu_entidad>  --project <nombre_proyecto>
```

Para más información sobre Barridos W&B, consulta el capítulo [Ajustar Hiperparámetros](../sweeps/intro.md).


</TabItem>

</Tabs>

## Crear un programador de barridos personalizado
Crea un programador de barridos personalizado ya sea con el programador de W&B o un programador personalizado.

:::info
Usar trabajos de programador requiere la versión de la CLI de wandb >= `0.15.4`
:::

<Tabs
  defaultValue="programador-wandb"
  values={[
    {label: 'Programador Wandb', value: 'wandb-scheduler'},
    {label: 'Programador Optuna', value: 'optuna-scheduler'},
    {label: 'Programador personalizado', value: 'custom-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

  Crea un barrido de lanzamiento utilizando la lógica de programación de barridos de W&B como un trabajo.
  
  1. Identifica el trabajo del programador Wandb en el proyecto público wandb/sweep-jobs, o usa el nombre del trabajo:
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. Construye un yaml de configuración con un bloque adicional `scheduler` que incluya una clave `job` apuntando a este nombre, ejemplo abajo.
  3. Usa el comando `wandb launch-sweep` con la nueva configuración.


Ejemplo de configuración:
```yaml
# launch-sweep-config.yaml  
description: Configuración de barrido de lanzamiento usando un trabajo de programador
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # permite 8 ejecuciones de barrido concurrentes

# trabajo de entrenamiento/ajuste que las ejecuciones de barrido llevarán a cabo
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```

  </TabItem>
  <TabItem value="custom-scheduler">

  Los programadores personalizados se pueden crear creando un trabajo de programador. Para los propósitos de esta guía, modificaremos el `WandbScheduler` para proporcionar más registros. 

  1. Clona el repositorio `wandb/launch-jobs` (específicamente: `wandb/launch-jobs/jobs/sweep_schedulers`)
  2. Ahora, podemos modificar el `wandb_scheduler.py` para lograr nuestro registro aumentado deseado. Ejemplo: Agregar registro a la función `_poll`. Esto se llama una vez en cada ciclo de sondeo (temporización configurable), antes de lanzar nuevas ejecuciones de barrido. 
  3. Ejecuta el archivo modificado para crear un trabajo, con: `python wandb_scheduler.py --project <proyecto> --entity <entidad> --name CustomWandbScheduler`
  4. Identifica el nombre del trabajo creado, ya sea en la interfaz de usuario o en la salida de la llamada anterior, que será un trabajo de artefacto de código (a menos que se especifique lo contrario).
  5. Ahora crea una configuración de barrido donde el programador apunte a tu nuevo trabajo!

```yaml
...
scheduler:
  job: '<entidad>/<proyecto>/job-CustomWandbScheduler:latest'
...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

  Optuna es un marco de optimización de hiperparámetros que utiliza una variedad de algoritmos para encontrar los mejores hiperparámetros para un modelo dado (similar a W&B). Además de los [algoritmos de muestreo](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html), Optuna también proporciona una variedad de [algoritmos de poda](https://optuna.readthedocs.io/en/stable/reference/pruners.html) que se pueden usar para terminar temprano ejecuciones de bajo rendimiento. Esto es especialmente útil al ejecutar un gran número de ejecuciones, ya que puede ahorrar tiempo y recursos. Las clases son altamente configurables, solo pasa los parámetros esperados en el bloque `scheduler.settings.pruner/sampler.args` del archivo de configuración.



Crea un barrido de lanzamiento utilizando la lógica de programación de Optuna con un trabajo.

1. Primero, crea tu propio trabajo o usa un trabajo de imagen de programador Optuna preconstruido. 
    * Consulta el repositorio [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) para ejemplos sobre cómo crear tu propio trabajo.
    * Para usar una imagen Optuna preconstruida, puedes navegar a `job-optuna-sweep-scheduler` en el proyecto `wandb/sweep-jobs` o puedes usar el nombre del trabajo: `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`. 
    

2. Después de crear un trabajo, ahora puedes crear un barrido. Construye una configuración de barrido que incluya un bloque `scheduler` con una clave `job` apuntando al trabajo de programador Optuna (ejemplo abajo).

```yaml
  # optuna_config_basic.yaml
  description: Un programador Optuna básico
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # requerido para trabajos de programador provenientes de imágenes
    num_workers: 2

    # configuraciones específicas de optuna
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # eliminar el 75% de las ejecuciones
          n_warmup_steps: 10  # poda deshabilitada para los primeros x pasos

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. Por último, lanza el barrido a una cola activa con el comando launch-sweep:
  
  ```bash
  wandb launch-sweep <config.yaml> -q <cola> -p <proyecto> -e <entidad>
  ```


  Para la implementación exacta del trabajo de programador de barridos Optuna, consulta [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py). Para más ejemplos de lo que es posible con el programador Optuna, consulta [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler).


  </TabItem>
</Tabs>

 Ejemplos de lo que es posible con trabajos de programador de barridos personalizados están disponibles en el repositorio [wandb/launch-jobs](https://github.com/wandb/launch-jobs) bajo `jobs/sweep_schedulers`. Esta guía muestra cómo usar el **Trabajo de Programador Wandb** disponible públicamente, y también demuestra un proceso para crear trabajos de programador de barridos personalizados. 


 ## Cómo reanudar barridos en lanzamiento
  También es posible reanudar un barrido-lanzamiento desde un barrido lanzado previamente. Aunque los hiperparámetros y el trabajo de entrenamiento no pueden cambiarse, los parámetros específicos del programador sí pueden, así como la cola a la que se envía.

:::info
Si el barrido inicial utilizó un trabajo de entrenamiento con un alias como 'último', reanudar puede llevar a resultados diferentes si la última versión del trabajo ha cambiado desde la última ejecución.
:::

  1. Identifica el nombre/ID del barrido para un barrido de lanzamiento previamente ejecutado. El ID del barrido es una cadena de ocho caracteres (por ejemplo, `hhd16935`) que puedes encontrar en tu proyecto en la Aplicación W&B.
  2. Si cambias los parámetros del programador, construye un archivo de configuración actualizado.
  3. En tu terminal, ejecuta el siguiente comando. Reemplaza el contenido envuelto en "<" y ">" con tu información: 

```bash
wandb launch-sweep <config.yaml opcional> --resume_id <id del barrido> --queue <nombre_cola>
```