---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Opciones de configuración de barrido

Una configuración de barrido consiste en pares clave-valor anidados. Usa claves de nivel superior dentro de tu configuración de barrido para definir cualidades de tu búsqueda de barrido, como los parámetros a buscar ([clave `parameter`](./sweep-config-keys.md#parameters)), la metodología para buscar en el espacio de parámetros ([clave `method`](./sweep-config-keys.md#method)), y más.

La siguiente tabla lista las claves de configuración de barrido de nivel superior y una breve descripción. Consulta las respectivas secciones para más información sobre cada clave.


| Claves de nivel superior    | Descripción                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `program`         | (requerido) Script de entrenamiento a ejecutar.                                                                                            |
| `entity`          | Especifica la entidad para este barrido.                                                                                            |
| `project`         | Especifica el proyecto para este barrido.                                                                                           |
| `description`     | Descripción de texto del barrido.                                                                                                |
| `name`            | El nombre del barrido, mostrado en la interfaz de W&B.                                                                               |
| [`method`](#method) | (requerido) Especifica la [estrategia de búsqueda](./define-sweep-configuration.md#configuration-keys).                               |
| [`metric`](#metric) | Especifica la métrica a optimizar (solo usada por ciertas estrategias de búsqueda y criterios de detención).                              |
| [`parameters`](#parameters) | (requerido) Especifica los límites de [parámetros](define-sweep-configuration.md#parameters) a buscar.                         |
| [`early_terminate`](#early_terminate) | Especifica cualquier [criterio de detención temprana](./define-sweep-configuration.md#early_terminate).                                 |
| [`command`](#command)         | Especifica la [estructura de comando](./define-sweep-configuration.md#command) para invocar y pasar argumentos al script de entrenamiento. |
| `run_cap` | Especifica un número máximo de runs en un barrido.                                                                                          |

Consulta la [estructura de configuración de barrido](./sweep-config-keys.md) para más información sobre cómo estructurar tu configuración de barrido.

## `metric`

Usa la clave de configuración de barrido de nivel superior `metric` para especificar el nombre, el objetivo y la métrica objetivo a optimizar.

|Clave | Descripción |
| -------- | --------------------------------------------------------- |
| `name`   | Nombre de la métrica a optimizar.                           |
| `goal`   | `minimize` o `maximize` (El predeterminado es `minimize`).  |
| `target` | Valor objetivo para la métrica que estás optimizando. El barrido no crea nuevos runs cuando o si un run alcanza un valor objetivo que especificas. Los agentes activos que tienen un run ejecutándose (cuando el run alcanza el objetivo) esperan hasta que el run se complete antes de que el agente deje de crear nuevos runs. |

## `parameters`
En tu archivo YAML o script de Python, especifica `parameters` como una clave de nivel superior. Dentro de la clave `parameters`, proporciona el nombre de un hiperparámetro que quieras optimizar. Los hiperparámetros comunes incluyen: tasa de aprendizaje, tamaño de lote, epochs, optimizadores, y más. Para cada hiperparámetro que definas en tu configuración de barrido, especifica una o más restricciones de búsqueda.

La siguiente tabla muestra las restricciones de búsqueda de hiperparámetros soportadas. Basado en tu hiperparámetro y caso de uso, usa una de las restricciones de búsqueda a continuación para indicarle a tu agente de barrido dónde (en el caso de una distribución) o qué (`value`, `values`, y así sucesivamente) buscar o usar.


| Restricción de búsqueda | Descripción   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | Especifica todos los valores válidos para este hiperparámetro. Compatible con `grid`.    |
| `value`         | Especifica el único valor válido para este hiperparámetro. Compatible con `grid`.  |
| `distribution`  | Especifica una [distribución](#distribution-options-for-random-and-bayesian-search) de probabilidad. Consulta la nota después de esta tabla para información sobre valores predeterminados. |
| `probabilities` | Especifica la probabilidad de seleccionar cada elemento de `values` al usar `random`.  |
| `min`, `max`    | (`int`o `float`) Valores máximo y mínimo. Si es `int`, para hiperparámetros distribuidos `int_uniform`. Si es `float`, para hiperparámetros distribuidos `uniform`. |
| `mu`            | (`float`) Parámetro de media para hiperparámetros distribuidos `normal` o `lognormal`. |
| `sigma`         | (`float`) Parámetro de desviación estándar para hiperparámetros distribuidos `normal` o `lognormal`. |
| `q`             | (`float`) Tamaño del paso de cuantización para hiperparámetros cuantizados.     |
| `parameters`    | Anida otros parámetros dentro de un parámetro de nivel raíz.    |


:::info
W&B establece las siguientes distribuciones basadas en las siguientes condiciones si una [distribución](#distribution-options-for-random-and-bayesian-search) no se especifica:
* `categorical` si especificas `values`
* `int_uniform` si especificas `max` y `min` como enteros
* `uniform` si especificas `max` y `min` como flotantes
* `constant` si proporcionas un conjunto a `value`
:::

## `method`
Especifica la estrategia de búsqueda de hiperparámetros con la clave `method`. Hay tres estrategias de búsqueda de hiperparámetros para elegir: grid, random y búsqueda bayesiana.

#### Búsqueda de grid
Itera sobre cada combinación de valores de hiperparámetros. La búsqueda de grid toma decisiones no informadas sobre el conjunto de valores de hiperparámetros a usar en cada iteración. La búsqueda de grid puede ser costosa computacionalmente.

La búsqueda de grid se ejecuta para siempre si está buscando dentro de un espacio de búsqueda continuo.

#### Búsqueda aleatoria
Elige un conjunto aleatorio y no informado de valores de hiperparámetros en cada iteración basado en una distribución. La búsqueda aleatoria se ejecuta para siempre a menos que detengas el proceso desde la línea de comando, dentro de tu script de python, o [la interfaz de usuario de W&B](./sweeps-ui.md).

Especifica el espacio de distribución con la clave métrica si eliges búsqueda aleatoria (`method: random`).

#### Búsqueda bayesiana
En contraste con la búsqueda [aleatoria](#random-search) y de [grid](#grid-search), los modelos bayesianos toman decisiones informadas. La optimización bayesiana usa un modelo probabilístico para decidir qué valores usar a través de un proceso iterativo de probar valores en una función sustituta antes de evaluar la función objetivo. La búsqueda bayesiana funciona bien para un pequeño número de parámetros continuos pero escala mal. Para más información sobre la búsqueda bayesiana, consulta el [paper de Introducción a la Optimización Bayesiana](https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf).

La búsqueda bayesiana se ejecuta para siempre a menos que detengas el proceso desde la línea de comando, dentro de tu script de python, o [la interfaz de usuario de W&B](./sweeps-ui.md).

### Opciones de distribución para búsqueda aleatoria y bayesiana
Dentro de la clave `parameter`, anida el nombre del hiperparámetro. A continuación, especifica la clave `distribution` y especifica una distribución para el valor.

Las siguientes tablas listan las distribuciones que W&B soporta.

| Valor para la clave `distribution`  | Descripción            |
| ------------------------ | ------------------------------------ |
| `constant`               | Distribución constante. Debes especificar el valor constante (`value`) a usar.                    |
| `categorical`            | Distribución categórica. Debes especificar todos los valores válidos (`values`) para este hiperparámetro. |
| `int_uniform`            | Distribución uniforme discreta en enteros. Debes especificar `max` y `min` como enteros.     |
| `uniform`                | Distribución uniforme continua. Debes especificar `max` y `min` como flotantes.      |
| `q_uniform`              | Distribución uniforme cuantizada. Devuelve `round(X / q) * q` donde X es uniforme. `q` se predetermina en `1`.|
| `log_uniform`            | Distribución log-uniforme. Devuelve un valor `X` entre `exp(min)` y `exp(max)`tal que el logaritmo natural está uniformemente distribuido entre `min` y `max`.   |
| `log_uniform_values`     | Distribución log-uniforme. Devuelve un valor `X` entre `min` y `max` tal que `log(X)` está uniformemente distribuido entre `log(min)` y `log(max)`.     |
| `q_log_uniform`          | Log uniforme cuantizado. Devuelve `round(X / q) * q` donde `X` es `log_uniform`. `q` se predetermina en `1`. |
| `q_log_uniform_values`   | Log uniforme cuantizado. Devuelve `round(X / q) * q` donde `X` es `log_uniform_values`. `q` se predetermina en `1`.  |
| `inv_log_uniform`        | Distribución uniforme log inversa. Devuelve `X`, donde  `log(1/X)` está uniformemente distribuido entre `min` y `max`. |
| `inv_log_uniform_values` | Distribución uniforme log inversa. Devuelve `X`, donde  `log(1/X)` está uniformemente distribuido entre `log(1/max)` y `log(1/min)`.    |
| `normal`                 | Distribución normal. El valor de retorno está distribuido normalmente con media `mu` (predeterminado `0`) y desviación estándar `sigma` (predeterminado `1`).|
| `q_normal`               | Distribución normal cuantizada. Devuelve `round(X / q) * q` donde `X` es `normal`. Q se predetermina en 1.  |
| `log_normal`             | Distribución log normal. Devuelve un valor `X` tal que el logaritmo natural `log(X)` está distribuido normalmente con media `mu` (predeterminado `0`) y desviación estándar `sigma` (predeterminado `1`). |
| `q_log_normal`  | Distribución log normal cuantizada. Devuelve `round(X / q) * q` donde `X` es `log_normal`. `q` se predetermina en `1`. |

## `early_terminate`

Usa la terminación temprana (`early_terminate`) para detener runs con bajo rendimiento. Si ocurre una terminación temprana, W&B detiene el run actual antes de crear un nuevo run con un nuevo conjunto de valores de hiperparámetros.

:::note
Debes especificar un algoritmo de detención si usas `early_terminate`. Anida la clave `type` dentro de `early_terminate` dentro de tu configuración de barrido.
:::

### Algoritmo de detención

:::info
W&B actualmente soporta el algoritmo de detención [Hyperband](https://arxiv.org/abs/1603.06560). 
:::

El algoritmo de optimización de hiperparámetros [Hyperband](https://arxiv.org/abs/1603.06560) evalúa si un programa debe detenerse o si debe continuar en uno o más conteos de iteración preestablecidos, llamados *brackets*.

Cuando un run de W&B alcanza un bracket, el barrido compara la métrica de ese run con todos los valores de métrica previamente reportados. El barrido termina el run si el valor de la métrica del run es demasiado alto (cuando el objetivo es la minimización) o si la métrica del run es demasiado baja (cuando el objetivo es la maximización).

Los brackets se basan en el número de iteraciones registradas. El número de brackets corresponde al número de veces que registras la métrica que estás optimizando. Las iteraciones pueden corresponder a pasos, epochs, o algo intermedio. El valor numérico del contador de paso no se usa en los cálculos de brackets.

:::info
Especifica `min_iter` o `max_iter` para crear un horario de brackets.
:::


| Clave        | Descripción                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | Especifica la iteración para el primer bracket                    |
| `max_iter` | Especifica el número máximo de iteraciones.                      |
| `s`        | Especifica el número total de brackets (requerido para `max_iter`) |
| `eta`      | Especifica el horario multiplicador del bracket (predeterminado: `3`).        |
| `strict`   | Habilita el modo 'strict' que poda runs agresivamente, siguiendo más de cerca el paper original de Hyperband. Predeterminado en falso. |



:::info
Hyperband verifica qué [runs de W&B](../../ref/python/run.md) terminar una vez cada pocos minutos. La marca de tiempo de fin de run podría diferir de los brackets especificados si tu run o iteración son cortos.
:::

## `command` 

<!-- Agents created with [`wandb agent`](../../ref/cli/wandb-agent.md) receive a command in the following format by default: -->

Modifica el formato y el contenido con valores anidados dentro de la clave `command`. Puedes incluir directamente componentes fijos como nombres de archivo.

:::info
En sistemas Unix, `/usr/bin/env` asegura que el sistema operativo elija el intérprete de Python correcto basado en el entorno.
:::

W&B soporta las siguientes macros para componentes variables del comando:

| Macro de comando              | Descripción                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | `/usr/bin/env` en sistemas Unix, omitido en Windows.                                                                                                                   |
| `${interpreter}`           | Se expande a `python`.                                                                                                                                                  |
| `${program}`               | Nombre de archivo del script de entrenamiento especificado por la clave `program` de la configuración de barrido.                                                                                          |
| `${args}`                  | Hiperparámetros y sus valores en la forma `--param1=value1 --param2=value2`.                                                                                       |
| `${args_no_boolean_flags}` | Hiperparámetros y sus valores en la forma `--param1=value1` excepto los parámetros booleanos están en la forma `--boolean_flag_param` cuando es `True` y omitido cuando es `False`. |
| `${args_no_hyphens}`       | Hiperparámetros y sus valores en la forma `param1=value1 param2=value2`.                                                                                           |
| `${args_json}`             | Hiperparámetros y sus valores codificados como JSON.                                                                                                                     |
| `${args_json_file}`        | La ruta a un archivo que contiene los hiperparámetros y sus valores codificados como JSON.                                                                                   |
| `${envvar}`                | Una manera de pasar variables de entorno. `${envvar:MYENVVAR}` __ se expande al valor de la variable de entorno MYENVVAR. __                                               |