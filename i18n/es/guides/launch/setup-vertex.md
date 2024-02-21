---
displayed_sidebar: default
---

# Configuración de Vertex AI

Puedes usar W&B Launch para enviar trabajos para su ejecución como trabajos de entrenamiento de Vertex AI. Con los trabajos de entrenamiento de Vertex AI, puedes entrenar modelos de aprendizaje automático utilizando algoritmos proporcionados o personalizados en la plataforma de Vertex AI. Una vez que se inicia un trabajo de lanzamiento, Vertex AI gestiona la infraestructura subyacente, escalado y orquestación.

W&B Launch trabaja con Vertex AI a través de la clase `CustomJob` en el SDK de `google-cloud-aiplatform`. Los parámetros de un `CustomJob` pueden ser controlados con la configuración de la cola de lanzamiento. Vertex AI no puede ser configurado para extraer imágenes de un registro privado fuera de GCP. Esto significa que debes almacenar imágenes de contenedores en GCP o en un registro público si quieres usar Vertex AI con W&B Launch. Consulta la documentación de Vertex AI para más información sobre cómo hacer accesibles las imágenes de contenedores a los trabajos de Vertex.

<!-- Diagrama de Componentes de Launch en Vertex AI -->

## Prerrequisitos

1. **Crear o acceder a un proyecto de GCP con la API de Vertex AI habilitada.** Consulta la [documentación de la Consola de APIs de GCP](https://support.google.com/googleapi/answer/6158841?hl=en) para más información sobre cómo habilitar una API.
2. **Crear un repositorio de Registro de Artefactos de GCP** para almacenar imágenes que quieras ejecutar en Vertex. Consulta la [documentación del Registro de Artefactos de GCP](https://cloud.google.com/artifact-registry/docs/overview) para más información.
3. **Crear un bucket de GCS de staging** para que Vertex AI almacene sus metadatos. Ten en cuenta que este bucket debe estar en la misma región que tus cargas de trabajo de Vertex AI para poder ser usado como un bucket de staging. El mismo bucket puede ser usado para contextos de staging y de construcción.
4. **Crear una cuenta de servicio** con los permisos necesarios para iniciar trabajos de Vertex AI. Consulta la [documentación de GCP IAM](https://cloud.google.com/iam/docs/creating-managing-service-accounts) para más información sobre la asignación de permisos a cuentas de servicio.
5. **Otorgar a tu cuenta de servicio permiso para gestionar trabajos de Vertex**

|    Permiso    |    Alcance del Recurso     |      Descripción      | 
| ---------------- | --------------------- | --------------------- |
| `ml.jobs.create` | Proyecto de GCP Especificado | Permite la creación de nuevos trabajos de aprendizaje automático dentro del proyecto.    |
| `ml.jobs.list`   | Proyecto de GCP Especificado | Permite la lista de trabajos de aprendizaje automático dentro del proyecto.  |
| `ml.jobs.get`    | Proyecto de GCP Especificado | Permite la recuperación de información sobre trabajos de aprendizaje automático específicos dentro del proyecto. |

:::info
Si quieres que tus cargas de trabajo de Vertex AI asuman la identidad de una cuenta de servicio no estándar, consulta la documentación de Vertex AI para instrucciones sobre la creación de cuentas de servicio y los permisos necesarios. El campo `spec.service_account` de la configuración de la cola de lanzamiento puede ser usado para seleccionar una cuenta de servicio personalizada para tus runs de W&B.
:::

## Configurar una cola para Vertex AI
La configuración de la cola para los recursos de Vertex AI especifica las entradas al constructor `CustomJob` en el SDK de Python de Vertex AI, y al método `run` del `CustomJob`. Las configuraciones de recursos se almacenan bajo las claves `spec` y `run`: 

- La clave `spec` contiene valores para los argumentos nombrados del [constructor `CustomJob`](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec) en el SDK de Python de Vertex AI.
- La clave `run` contiene valores para los argumentos nombrados del método `run` de la clase `CustomJob` en el SDK de Python de Vertex AI.

Las personalizaciones del entorno de ejecución ocurren principalmente en la lista `spec.worker_pool_specs`. Una especificación de grupo de trabajadores define un grupo de trabajadores que ejecutarán tu trabajo. La especificación de trabajadores en la configuración predeterminada solicita una única máquina `n1-standard-4` sin aceleradores. Puedes cambiar el tipo de máquina, el tipo de acelerador y la cantidad según tus necesidades.

Para más información sobre los tipos de máquinas y tipos de aceleradores disponibles, consulta la [documentación de Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec).

## Crear una cola

Crea una cola en la App de W&B que use Vertex AI como su recurso de cómputo:

1. Navega a la [página de Launch](https://wandb.ai/launch).
2. Haz clic en el botón **Crear Cola**.
3. Selecciona la **Entidad** en la que te gustaría crear la cola.
4. Proporciona un nombre para tu cola en el campo **Nombre**.
5. Selecciona **GCP Vertex** como el **Recurso**.
6. Dentro del campo **Configuración**, proporciona información sobre tu `CustomJob` de Vertex AI que definiste en la sección anterior. Por defecto, W&B llenará un cuerpo de solicitud YAML y JSON similar al siguiente:
  ```yaml
  spec:
    worker_pool_specs:
      - machine_spec:
          machine_type: n1-standard-4
          accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
          accelerator_count: 0
        replica_count: 1
        container_spec:
          image_uri: ${image_uri}
    staging_bucket: <REQUIRED>
  run:
    restart_job_on_worker_restart: false
  ```
7. Después de configurar tu cola, haz clic en el botón **Crear Cola**.

Debes especificar al menos:
* `spec.worker_pool_specs` : lista no vacía de especificaciones de grupo de trabajadores.
* `spec.staging_bucket` : bucket de GCS que se utilizará para el staging de activos y metadatos de Vertex AI.

:::caution
Algunos de los documentos de Vertex AI muestran especificaciones de grupo de trabajadores con todas las claves en camel case, por ejemplo, `workerPoolSpecs`. El SDK de Python de Vertex AI usa snake case para estas claves, por ejemplo `worker_pool_specs`. 

Cada clave en la configuración de la cola de lanzamiento debe usar snake case.
:::

## Configurar un agente de lanzamiento
El agente de lanzamiento es configurable a través de un archivo de configuración que, por defecto, se encuentra en `~/.config/wandb/launch-config.yaml`.

```yaml
max_jobs: <n-trabajos-concurrentes>
queues:
  - <nombre-de-la-cola>
```

Si tienes la intención de que el agente de lanzamiento construya imágenes para ser ejecutadas en Vertex AI, consulta Configuración avanzada del agente.

Si quieres que el agente de lanzamiento construya imágenes para ti que se ejecuten en Vertex AI, consulta [Configuración avanzada del agente](./setup-agent-advanced.md).

## Configurar permisos del agente
Hay varios métodos para autenticarse como esta cuenta de servicio. Esto se puede lograr a través de la Identidad de Carga de Trabajo, un JSON de cuenta de servicio descargado, variables de entorno, la herramienta de línea de comandos de Google Cloud Platform, o una combinación de estos métodos.