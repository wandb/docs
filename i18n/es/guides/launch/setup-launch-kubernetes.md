---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Configurar Kubernetes

Puedes usar W&B Launch para ejecutar trabajos de lanzamiento de W&B como un [Trabajo de Kubernetes](https://kubernetes.io/docs/concepts/workloads/controllers/job/) o recurso de [Carga de trabajo personalizada](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) en un cluster de Kubernetes. Esto es particularmente útil si deseas usar Kubernetes para gestionar tu cluster de computación y quieres una interfaz simple para ejecutar entrenamiento, transformación o cargas de trabajo de ML en tu cluster.

W&B mantiene una [imagen oficial del agente de lanzamiento](https://hub.docker.com/r/wandb/launch-agent) que puede ser desplegada en tu cluster con un [gráfico de helm](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) que es gestionado por W&B.

:::info
Es posible iniciar un agente de lanzamiento fuera de un cluster de Kubernetes. Sin embargo, recomendamos que despliegues el agente de lanzamiento directamente en el cluster de Kubernetes que es responsable de ejecutar el Trabajo de Kubernetes o la Carga de trabajo personalizada.
:::

El agente de lanzamiento envía cargas de trabajo al cluster especificado por el contexto actual del cluster de Kubernetes.

W&B usa el constructor [Kaniko](https://github.com/GoogleContainerTools/kaniko) para permitir que el agente de lanzamiento construya imágenes de Docker en un cluster de Kubernetes. Para aprender más sobre cómo configurar Kaniko para el agente de lanzamiento, consulta [Configuración avanzada del agente](./setup-agent-advanced.md).

<!-- Futuro: insertar diagrama aquí -->

## Configurar una cola para Kubernetes

La configuración de la cola de lanzamiento para un recurso objetivo de Kubernetes se parecerá a una especificación de Trabajo de Kubernetes o a una especificación de Recurso Personalizado de Kubernetes. Puedes controlar cualquier aspecto de la especificación del recurso de carga de trabajo de Kubernetes cuando creas una cola de lanzamiento.

<Tabs
defaultValue="job"
values={[
{label: 'Especificación de Trabajo de Kubernetes', value: 'job'},
{label: 'Especificación de Recurso Personalizado', value: 'custom'},
]}>

<TabItem value="job">

```yaml
spec:
  template:
    spec:
      containers:
        - env:
            - name: MY_ENV_VAR
              value: some-value
          resources:
            requests:
              cpu: 1000m
              memory: 1Gi
metadata:
  labels:
    queue: k8s-test
namespace: wandb
```

</TabItem>
<TabItem value="custom">

En algunos casos de uso, podrías querer usar definiciones de `CustomResource`. Las definiciones de `CustomResource` son útiles si, por ejemplo, quieres realizar entrenamiento distribuido en múltiples nodos. Consulta el tutorial para usar Launch con trabajos de múltiples nodos utilizando Volcano para un ejemplo de aplicación. Otro caso de uso podría ser que quieras usar W&B Launch con Kubeflow.

El siguiente fragmento de YAML muestra una muestra de configuración de cola de lanzamiento que usa Kubeflow:

```yaml
kubernetes:
  kind: PyTorchJob
  spec:
    pytorchReplicaSpecs:
      Master:
        replicas: 1
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
      Worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
    ttlSecondsAfterFinished: 600
  metadata:
    name: '${run_id}-pytorch-job'
  apiVersion: kubeflow.org/v1
```

  </TabItem>
</Tabs>

Por razones de seguridad, W&B inyectará los siguientes recursos en tu cola de lanzamiento si no están especificados:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

El siguiente fragmento de YAML demuestra cómo aparecerán estos valores en tu cola de lanzamiento:

```yaml title="example-spec.yaml"
spec:
  template:
    `backOffLimit`: 0
    ttlSecondsAfterFinished: 60
    securityContext:
      allowPrivilegeEscalation: False,
      capabilities:
        drop:
          - ALL,
      seccompProfile:
        type: "RuntimeDefault"
```

## Crear una cola

Crea una cola en la aplicación W&B que usa Kubernetes como su recurso de computación:

1. Navega a la [página de lanzamiento](https://wandb.ai/launch).
2. Haz clic en el botón **Crear cola**.
3. Selecciona la **Entidad** en la que te gustaría crear la cola.
4. Proporciona un nombre para tu cola en el campo **Nombre**.
5. Selecciona **Kubernetes** como el **Recurso**.
6. Dentro del campo **Configuración**, proporciona la especificación de flujo de trabajo de Trabajo de Kubernetes o Especificación de Recurso Personalizado que [configuraste en la sección anterior](#configure-a-queue-for-kubernetes).

## Configurar un agente de lanzamiento con helm

Usa el gráfico de helm proporcionado por W&B para desplegar el agente de lanzamiento en tu cluster de Kubernetes. Controla el comportamiento del agente de lanzamiento con el archivo `values.yaml` [file](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml).

Especifica los contenidos que normalmente serían definidos en tu archivo de configuración del agente de lanzamiento (`~/.config/wandb/launch-config.yaml`) dentro de la clave `launchConfig` en el archivo `values.yaml`.

Por ejemplo, supongamos que tienes una configuración del agente de lanzamiento que te permite ejecutar un agente de lanzamiento en EKS que usa el constructor de imágenes Docker de Kaniko:

```yaml title="launch-config.yaml"
queues:
	- <nombre de cola>
max_jobs: <n trabajos concurrentes>
environment:
	type: aws
	region: us-east-1
registry:
	type: ecr
	uri: <mi-uri-de-registry>
builder:
	type: kaniko
	build-context-store: <uri-de-bucket-s3>
```

Dentro de tu archivo `values.yaml`, esto podría parecer:

```yaml title="values.yaml"
agent:
  labels: {}
  # Clave API de W&B.
  apiKey: ''
  # Imagen del contenedor para usar en el agente.
  image: wandb/launch-agent:latest
  # Política de extracción de imagen para la imagen del agente.
  imagePullPolicy: Always
  # Bloque de recursos para la especificación del agente.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Namespace para desplegar el agente de lanzamiento
namespace: wandb

# URL de la API de W&B (Configura la tuya aquí)
baseUrl: https://api.wandb.ai

# Namespaces objetivo adicionales en los que el agente de lanzamiento puede desplegar
additionalTargetNamespaces:
  - default
  - wandb

# Esto debe ser configurado con el contenido literal de tu configuración del agente de lanzamiento.
launchConfig: |
  queues:
    - <nombre de cola>
  max_jobs: <n trabajos concurrentes>
  environment:
    type: aws
    region: <región-aws>
  registry:
    type: ecr
    uri: <mi-uri-de-registry>
  builder:
    type: kaniko
    build-context-store: <uri-de-bucket-s3>

# El contenido de un archivo de credenciales de git. Esto será almacenado en un secreto de k8s
# y montado en el contenedor del agente. Configura esto si quieres clonar repos privados.
gitCreds: |

# Anotaciones para la cuenta de servicio de wandb. Útil al configurar la identidad de carga de trabajo en gcp.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# Configura para la clave de acceso a almacenamiento de azure si usas kaniko con azure.
azureStorageAccessKey: ''
```

Para más información sobre registros, entornos y permisos requeridos del agente consulta [Configuración avanzada del agente](./setup-agent-advanced.md).

Sigue las instrucciones en el [repositorio del gráfico de helm](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) para desplegar tu agente.