---
description: Answers to frequently asked question about W&B Sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# FAQ

<head>
  <title>Preguntas Frecuentes Sobre Barridos</title>
</head>

### ¿Necesito proporcionar valores para todos los hiperparámetros como parte del Barrido de W&B? ¿Puedo establecer valores predeterminados?

Los nombres y valores de los hiperparámetros especificados como parte de la configuración del barrido son accesibles en `wandb.config`, un objeto similar a un diccionario.

Para runs que no son parte de un barrido, los valores de `wandb.config` se establecen usualmente proporcionando un diccionario al argumento `config` de `wandb.init`. Durante un barrido, sin embargo, cualquier información de configuración pasada a `wandb.init` se trata en cambio como un valor predeterminado, que podría ser sobrescrito por el barrido.

También puedes ser más explícito sobre el comportamiento deseado utilizando `config.setdefaults`. Los fragmentos de código para ambos métodos aparecen a continuación:

<Tabs
  defaultValue="wandb.init"
  values={[
    {label: 'wandb.init', value: 'wandb.init'},
    {label: 'config.setdefaults', value: 'config.setdef'},
  ]}>
  <TabItem value="wandb.init">

```python
# establecer valores predeterminados para hiperparámetros
config_defaults = {"lr": 0.1, "batch_size": 256}

# iniciar un run, proporcionando valores predeterminados
#   que pueden ser sobrescritos por el barrido
with wandb.init(config=config_default) as run:
    # añade aquí tu código de entrenamiento
    ...
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# establecer valores predeterminados para hiperparámetros
config_defaults = {"lr": 0.1, "batch_size": 256}

# iniciar un run
with wandb.init() as run:
    # actualizar cualquier valor no establecido por el barrido
    run.config.setdefaults(config_defaults)

    # añade aquí tu código de entrenamiento
```

  </TabItem>
</Tabs>

### ¿Cómo debería ejecutar barridos en SLURM?

Cuando usas barridos con el [sistema de programación SLURM](https://slurm.schedmd.com/documentation.html), recomendamos ejecutar `wandb agent --count 1 SWEEP_ID` en cada uno de tus trabajos programados, lo que ejecutará un único trabajo de entrenamiento y luego saldrá. Esto facilita predecir tiempos de ejecución al solicitar recursos y aprovecha el paralelismo de la búsqueda de hiperparámetros.

### ¿Puedo volver a ejecutar una búsqueda en cuadrícula?

Sí. Si agotas una búsqueda en cuadrícula pero quieres reejecutar algunos de los Runs de W&B (por ejemplo, porque algunos se bloquearon). Elimina los Runs de W&B que quieras reejecutar, luego elige el botón **Resume** en la [página de control del barrido](./sweeps-ui.md). Finalmente, inicia nuevos agentes de Barrido de W&B con el nuevo ID del Barrido.

Las combinaciones de parámetros con Runs de W&B completados no se vuelven a ejecutar.

### ¿Cómo uso comandos CLI personalizados con barridos?

Puedes usar Barridos de W&B con comandos CLI personalizados si normalmente configuras algunos aspectos del entrenamiento pasando argumentos de línea de comandos.

Por ejemplo, el siguiente fragmento de código demuestra un terminal bash donde el usuario está entrenando un script de Python llamado train.py. El usuario pasa valores que luego se analizan dentro del script de Python:

```bash
/usr/bin/env python train.py -b \
    tu-configuración-de-entrenamiento \
    --batchsize 8 \
    --lr 0.00001
```

Para usar comandos personalizados, edita la clave `command` en tu archivo YAML. Por ejemplo, continuando con el ejemplo anterior, podría verse así:

```yaml
program:
  train.py
method: grid
parameters:
  batch_size:
    value: 8
  lr:
    value: 0.0001
command:
  - ${env}
  - python
  - ${program}
  - "-b"
  - tu-configuración-de-entrenamiento
  - ${args}
```

La clave `${args}` se expande a todos los parámetros en el archivo de configuración del barrido, expandidos para que puedan ser analizados por `argparse: --param1 valor1 --param2 valor2`

Si tienes argumentos adicionales que no quieres especificar con `argparse` puedes usar:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::info
Dependiendo del entorno, `python` podría apuntar a Python 2. Para asegurar que se invoque Python 3, usa `python3` en lugar de `python` al configurar el comando:

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
:::

### ¿Hay alguna manera de agregar valores extra a un barrido, o necesito comenzar uno nuevo?

No puedes cambiar la configuración del Barrido una vez que un Barrido de W&B ha comenzado. Pero puedes ir a cualquier vista de tabla, y usar las casillas de verificación para seleccionar runs, luego usar la opción de menú **Create sweep** para crear una nueva configuración de Barrido usando runs anteriores.

### ¿Podemos marcar variables booleanas como hiperparámetros?

Puedes usar la macro `${args_no_boolean_flags}` en la sección de comando de la configuración para pasar hiperparámetros como flags booleanos. Esto pasará automáticamente cualquier parámetro booleano como un flag. Cuando `param` es `True` el comando recibirá `--param`, cuando `param` es `False` el flag será omitido.

### ¿Puedo usar Barridos y SageMaker?

Sí. A grandes rasgos, necesitarás autenticar W&B y necesitarás crear un archivo `requirements.txt` si usas un estimador incorporado de SageMaker. Para más información sobre cómo autenticar y configurar un archivo requirements.txt, consulta la [guía de integración de SageMaker](../integrations/other/sagemaker.md).

:::info
Un ejemplo completo está disponible en [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) y puedes leer más en nuestro [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker).\
También puedes leer el [tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) sobre cómo desplegar un analizador de sentimientos usando SageMaker y W&B.
:::

### ¿Puedes usar Barridos de W&B con infraestructuras en la nube como AWS Batch, ECS, etc.?

En general, necesitarías una manera de publicar `sweep_id` en una ubicación que cualquier potencial agente de Barrido de W&B pueda leer y una manera para que estos agentes de Barrido consuman este `sweep_id` y comiencen a ejecutar.

En otras palabras, necesitas algo que pueda invocar `wandb agent`. Por ejemplo, levantar una instancia EC2 y luego llamar a `wandb agent` en ella. En este caso, podrías usar una cola SQS para transmitir `sweep_id` a algunas instancias EC2 y luego hacer que consuman el `sweep_id` de la cola y comiencen a ejecutar.

### ¿Cómo puedo cambiar el directorio donde mi barrido registra localmente?

Puedes cambiar la ruta del directorio donde W&B registrará los datos de tu run estableciendo una variable de entorno `WANDB_DIR`. Por ejemplo:

```python
os.environ["WANDB_DIR"] = os.path.abspath("tu/directorio")
```

### Optimización de múltiples métricas

Si deseas optimizar múltiples métricas en el mismo run, puedes usar una suma ponderada de las métricas individuales.

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

Asegúrate de registrar tu nueva métrica combinada y establecerla como el objetivo de optimización:

```yaml
metric:
  name: metric_combined
  goal: minimize
```

### ¿Cómo habilito el registro de código con Barridos?

Para habilitar el registro de código para barridos, simplemente añade `wandb.log_code()` después de haber inicializado tu Run de W&B. Esto es necesario incluso cuando has habilitado el registro de código en la página de configuración de tu perfil de W&B en la aplicación. Para un registro de código más avanzado, consulta la [documentación de `wandb.log_code()` aquí](../../ref/python/run.md#log_code).

### ¿Qué es la columna "Est. Runs"?

W&B proporciona un número estimado de Runs que ocurrirán cuando creas un Barrido de W&B con un espacio de búsqueda discreto. El número total de Runs es el producto cartesiano del espacio de búsqueda.

Por ejemplo, supongamos que proporcionas el siguiente espacio de búsqueda:

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

El producto cartesiano en este ejemplo es 9. W&B muestra este número en la UI de la App de W&B como el conteo estimado de runs (**Est. Runs**):

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)


También puedes obtener el conteo estimado de Runs con el SDK de W&B. Utiliza el atributo `expected_run_count` del objeto Sweep para obtener el conteo estimado de Runs:

```python
sweep_id = wandb.sweep(
    sweep_configs, project="tu_nombre_de_proyecto", entity="tu_nombre_de_entidad"
)
api = wandb.Api()
sweep = api.sweep(f"tu_nombre_de_entidad/tu_nombre_de_proyecto/sweeps/{sweep_id}")
print(f"CONTEO ESTIMADO DE RUNS = {sweep.expected_run_count}")
```