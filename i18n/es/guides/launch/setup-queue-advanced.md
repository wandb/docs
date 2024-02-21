---
displayed_sidebar: default
---

# Configuración avanzada de cola
La siguiente página describe cómo configurar opciones adicionales de la cola de lanzamiento.

## Configurar plantillas de configuración de cola
Administre y gestione límites en el consumo de cómputo con Plantillas de Configuración de Cola. Establezca valores predeterminados, mínimos y máximos para campos como el consumo de memoria, GPU y la duración de ejecución.

Después de configurar una cola con plantillas de configuración, los miembros de su equipo pueden alterar los campos que definió solo dentro del rango especificado por usted.

### Configurar plantilla de cola
Puede configurar una plantilla de cola en una cola existente o crear una nueva.

1. Navegue a la Aplicación de Lanzamiento en [https://wandb.ai/launch](https://wandb.ai/launch).
2. Seleccione **Ver cola** junto al nombre de la cola a la que desea agregar una plantilla.
3. Seleccione la pestaña **Config**. Esto mostrará información sobre su cola, como cuándo se creó la cola, la configuración de la cola y las anulaciones existentes en el momento del lanzamiento.
4. Navegue a la sección **Configuración de cola**.
5. Identifique las claves de configuración que desea crear una plantilla para.
6. Reemplace el valor en la configuración con un campo de plantilla. Los campos de plantilla toman la forma de `{{nombre-variable}}`.
7. Haga clic en el botón **Parsear configuración**. Cuando parsee su configuración, W&B creará automáticamente mosaicos debajo de la configuración de la cola para cada plantilla que creó.
8. Para cada mosaico generado, primero debe especificar el tipo de datos (cadena, entero o flotante) que la configuración de la cola puede permitir. Para hacer esto, seleccione el tipo de datos del menú desplegable **Tipo**.
9. Basado en su tipo de datos, complete los campos que aparecen dentro de cada mosaico.
10. Haga clic en **Guardar configuración**.

Por ejemplo, suponga que desea crear una plantilla que limite qué instancias de AWS puede usar su equipo. Antes de agregar un campo de plantilla, su configuración de cola podría verse algo similar a:

```yaml title="configuración de lanzamiento"
RoleArn: arn:aws:iam:región:id-cuenta:tipo-recurso/id-recurso
ResourceConfig:
  InstanceType: ml.m4.xlarge
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://nombrebucket
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

Cuando agrega un campo de plantilla para el `InstanceType`, su configuración se verá así:

```yaml title="configuración de lanzamiento"
RoleArn: arn:aws:iam:región:id-cuenta:tipo-recurso/id-recurso
ResourceConfig:
  InstanceType: "{{aws_instance}}"
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://nombrebucket
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

A continuación, haga clic en **Parsear configuración**. Un nuevo mosaico etiquetado como `aws-instance` aparecerá debajo de la **Configuración de cola**.

Desde allí, seleccione String como el tipo de datos del menú desplegable **Tipo**. Esto llenará campos donde puede especificar valores que un usuario puede elegir. Por ejemplo, en la siguiente imagen, el administrador del equipo configuró dos tipos diferentes de instancias de AWS que los usuarios pueden elegir (`ml.m4.xlarge` y `ml.p3.xlarge`):

![](/images/launch/aws_template_example.png)

## Configurar dinámicamente trabajos de lanzamiento
Las configuraciones de cola pueden configurarse dinámicamente utilizando macros que se evalúan cuando el agente desencola un trabajo de la cola. Puede establecer las siguientes macros:

| Macro             | Descripción                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | El nombre del proyecto al que se lanzará la ejecución. |
| `${entity_name}`  | El propietario del proyecto al que se lanzará la ejecución.   |
| `${run_id}`       | El id de la ejecución que se está lanzando.                     |
| `${run_name}`     | El nombre de la ejecución que se está lanzando.                |
| `${image_uri}`    | El URI de la imagen de contenedor para esta ejecución.          |

:::info
Cualquier macro personalizada no listada en la tabla anterior (por ejemplo, `${MY_ENV_VAR}`), se sustituye por una variable de entorno del entorno del agente.
:::

## Utilizar el agente de lanzamiento para construir imágenes que se ejecuten en aceleradores (GPU)
Es posible que necesite especificar una imagen base de acelerador si utiliza launch para construir imágenes que se ejecuten en un entorno de acelerador.

Esta imagen base de acelerador debe satisfacer los siguientes requisitos:

- Compatibilidad con Debian (el Dockerfile de Launch usa apt-get para obtener python)
- Conjunto de instrucciones de hardware de compatibilidad CPU & GPU (Asegúrese de que su versión de CUDA sea compatible con la GPU que tiene la intención de usar)
- Compatibilidad entre la versión del acelerador que proporciona y los paquetes instalados en su algoritmo de ML
- Paquetes instalados que requieren pasos adicionales para configurar la compatibilidad con el hardware

### Cómo utilizar GPUs con TensorFlow

Asegúrese de que TensorFlow utilice correctamente su GPU. Para lograr esto, especifique una imagen Docker y su etiqueta de imagen para la clave `builder.accelerator.base_image` en la configuración de recursos de la cola.

Por ejemplo, la imagen base `tensorflow/tensorflow:latest-gpu` asegura que TensorFlow utilice correctamente su GPU. Esto se puede configurar usando la configuración de recursos en la cola.

El siguiente fragmento de JSON demuestra cómo especificar la imagen base de TensorFlow en su configuración de cola:

```json title="Configuración de cola"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```