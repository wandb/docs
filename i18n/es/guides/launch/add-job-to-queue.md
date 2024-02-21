---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Añadir trabajos a la cola

La siguiente página describe cómo agregar trabajos de lanzamiento a una cola de lanzamiento.

:::info
Asegúrate de que tú, o alguien de tu equipo, ya haya configurado una cola de lanzamiento. Para más información, consulta la página de [Configuración de Lanzamiento](./setup-launch.md).
:::

## Añade trabajos a tu cola

Añade trabajos a tu cola interactivamente con la App de W&B o programáticamente con la CLI de W&B.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App de W&B', value: 'app'},
    {label: 'CLI de W&B', value: 'cli'},
  ]}>
  <TabItem value="app">
Añade un trabajo a tu cola programáticamente con la App de W&B.

1. Navega a la página de tu Proyecto de W&B.
2. Selecciona el icono de **Trabajos** en el panel izquierdo:
  ![](/images/launch/project_jobs_tab_gs.png)
3. La página de **Trabajos** muestra una lista de trabajos de lanzamiento de W&B que fueron creados a partir de ejecuciones de W&B anteriores. 
  ![](/images/launch/view_jobs.png)
4. Selecciona el botón de **Lanzamiento** junto al nombre del trabajo. Aparecerá un modal en el lado derecho de la página.
5. Del desplegable de **Versión del trabajo**, selecciona la versión del trabajo de lanzamiento que quieres usar. Los trabajos de lanzamiento se versionan como cualquier otro [Artefacto de W&B](../artifacts/create-a-new-artifact-version.md). Se crearán diferentes versiones del mismo trabajo de lanzamiento si haces modificaciones a las dependencias de software o al código fuente utilizado para ejecutar el trabajo.
6. Dentro de la sección de **Sobrescrituras**, proporciona nuevos valores para cualquier entrada que esté configurada para tu trabajo de lanzamiento. Las sobrescrituras comunes incluyen un nuevo comando de punto de entrada, argumentos o valores en el `wandb.config` de tu nueva ejecución de W&B.  
  ![](/images/launch/create_starter_queue_gs.png)
  Puedes copiar y pegar valores de otras ejecuciones de W&B que utilizaron tu trabajo de lanzamiento haciendo clic en el botón de **Pegar desde...**.
7. Del desplegable de **Cola**, selecciona el nombre de la cola de lanzamiento a la que quieres añadir tu trabajo de lanzamiento. 
8. Usa el desplegable de **Prioridad del trabajo** para especificar la prioridad de tu trabajo de lanzamiento. La prioridad de un trabajo de lanzamiento se establece en "Media" si la cola de lanzamiento no soporta priorización.
9. **(Opcional) Sigue este paso solo si un administrador de tu equipo creó una plantilla de configuración de cola**  
Dentro del campo de **Configuraciones de Cola**, proporciona valores para las opciones de configuración que fueron creadas por el administrador de tu equipo.  
Por ejemplo, en el siguiente ejemplo, el administrador del equipo configuró tipos de instancias de AWS que pueden ser utilizadas por el equipo. En este caso, los miembros del equipo pueden elegir entre el tipo de instancia de cómputo `ml.m4.xlarge` o `ml.p3.xlarge` para entrenar su modelo.
![](/images/launch/team_member_use_config_template.png)
10. Selecciona el **Proyecto de destino**, donde aparecerá la ejecución resultante. Este proyecto necesita pertenecer a la misma entidad que la cola.
11. Selecciona el botón de **Lanzar ahora**. 


  </TabItem>
    <TabItem value="cli">

Usa el comando `wandb launch` para añadir trabajos a una cola. Crea una configuración JSON con sobrescrituras de hiperparámetros. Por ejemplo, utilizando el script de la guía de [Inicio Rápido](./walkthrough.md), creamos un archivo JSON con las siguientes sobrescrituras:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },   
      "entry_point": []
  }
}
```

:::note
W&B Launch utilizará los parámetros predeterminados si no proporcionas un archivo de configuración JSON.
:::

Si quieres sobrescribir la configuración de la cola, o si tu cola de lanzamiento no tiene un recurso de configuración definido, puedes especificar la clave `resource_args` en tu archivo config.json. Por ejemplo, continuando con el ejemplo anterior, tu archivo config.json podría parecerse al siguiente:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },
      "entry_point": []
  },
  "resource_args": {
        "<tipo-de-recurso>" : {
            "<clave>": "<valor>"
        }
  }
}
```

Reemplaza los valores dentro de los `<>` con tus propios valores.



Proporciona el nombre de la cola para la bandera `queue`(`-q`), el nombre del trabajo para la bandera `job`(`-j`) y la ruta al archivo de configuración para la bandera `config`(`-c`).

```bash
wandb launch -j <trabajo> -q <nombre-de-cola> \ 
-e <nombre-de-entidad> -c path/to/config.json
```
Si trabajas dentro de un Equipo de W&B, sugerimos que especifiques la bandera de `entidad` (`-e`) para indicar qué entidad usará la cola.