---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Configuración avanzada del agente

Cómo configuras el agente de lanzamiento depende de numerosos factores. Uno de esos factores es si el agente de lanzamiento construye una imagen para ti o no.

:::tip
El agente de lanzamiento de W&B construye una imagen para ti si proporcionas un trabajo basado en un repositorio de Git o [trabajos basados en artefactos](./create-launch-job.md#create-a-job-with-a-wb-artifact).
:::

En el caso de uso más simple, proporcionas un trabajo de lanzamiento basado en imagen que se ejecuta en un entorno objetivo de la cola de lanzamiento que tiene acceso a tu repositorio de imágenes.

La siguiente sección describe los requisitos que debes cumplir si usas el agente de lanzamiento para construir imágenes para ti.

## Constructores

Los agentes de lanzamiento pueden construir imágenes a partir de artefactos de W&B y trabajos con origen en repositorios de Git. Esto significa que los ingenieros de ML pueden iterar rápidamente sobre el código sin necesidad de reconstruir imágenes de Docker ellos mismos. Para permitir este comportamiento del constructor, el archivo de configuración del agente de lanzamiento (`launch-config.yaml`) debe tener una opción de constructor especificada. W&B Launch admite dos constructores, Kaniko y Docker, junto con una opción `noop` que le indicará al agente que solo use imágenes preconstruidas.

* Kaniko: Usa Kaniko cuando el agente sondea colas de lanzamiento en un cluster de Kubernetes
* Docker: Usa Docker para todos los demás casos en los que quieras construir imágenes automáticamente.
* Noop: Usa cuando *solo* quieras usar imágenes preconstruidas. (Tanto los constructores Kaniko como Docker pueden usar imágenes preconstruidas o construir nuevas.)

### Docker

W&B recomienda que uses el constructor Docker si quieres que el agente construya imágenes en una máquina local (que tenga Docker instalado). Especifica el constructor Docker en la configuración del agente de lanzamiento con la clave de constructor.

Por ejemplo, el siguiente fragmento YAML muestra cómo especificar esto en un archivo de configuración del agente de lanzamiento (`launch-config.yaml`):

```yaml title="launch-config.yaml"
builder:
  type: docker
```

### Kaniko

Para usar el constructor Kaniko, debes especificar un registro de contenedores y una opción de entorno.

Por ejemplo, el siguiente fragmento YAML muestra cómo especificar Kaniko en un archivo de configuración del agente de lanzamiento (`launch-config.yaml`):

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/
  build-job-name: wandb-image-build # Nombre prefijo del trabajo de Kubernetes para todas las construcciones
```

Si ejecutas un cluster de Kubernetes diferente al uso de AKS, EKS o GKE, necesitas crear un secreto de Kubernetes que contenga las credenciales para tu entorno en la nube.

- Para otorgar acceso a GCP, este secreto debe contener un [JSON de cuenta de servicio](https://cloud.google.com/iam/docs/keys-create-delete#creating).
- Para otorgar acceso a AWS, este secreto debe contener un [archivo de credenciales de AWS](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html).

Dentro de tu archivo de configuración del agente, y dentro de la sección del constructor, establece las claves `secret-name` y `secret-key` para permitir que Kaniko use los secretos:

```yaml title="launch-config.yaml"
builder:
	type: kaniko
  build-context-store: <mi-almacenamiento-de-contexto-de-construcción>
  secret-name: <nombre-del-secreto-de-Kubernetes>
  secret-key: <nombre-del-archivo-secreto>
```

:::note
El constructor Kaniko requiere permisos para poner datos en el almacenamiento en la nube (como Amazon S3) consulta la sección [Permisos del agente](#agent-permissions) para obtener más información.
:::

## Conectar un agente a un registro de contenedores
Puedes conectar el agente de lanzamiento a un registro de contenedores como Amazon Elastic Container Registry (Amazon ECR), Google Artifact Registry en GCP o Azure Container Registry. Lo siguiente describe casos de uso comunes por los que podrías querer conectar el agente de lanzamiento a un registro de contenedores en la nube:

- no quieres almacenar imágenes que estás construyendo en tu máquina local
- quieres compartir imágenes en múltiples máquinas
- si el agente construye una imagen para ti y usas un recurso de cómputo en la nube como Amazon SageMaker o VertexAI.

Para conectar el agente de lanzamiento a un registro de contenedores, proporciona información adicional sobre el entorno y el registro que quieres usar en la configuración del agente de lanzamiento. Además, otorga al agente permisos dentro del entorno para interactuar con los componentes requeridos basados en tu caso de uso.


:::note
Los agentes de lanzamiento admiten *extraer* de cualquier registro de contenedores al que los nodos en los que se ejecuta el trabajo tengan acceso, incluyendo Dockerhub privado, JFrog, Quay, etc. Actualmente, *empujar* imágenes a registros solo se admite para ECR, ACR y GCR.
:::

### Configuración del agente

Dentro de tu configuración del agente de lanzamiento (`launch-config.yaml`), proporciona el nombre del entorno del recurso objetivo y el registro de contenedores para las claves `environment` y `registry`, respectivamente.

Las siguientes pestañas demuestran cómo configurar el agente de lanzamiento en función de tu entorno y registro.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

La configuración del entorno de AWS requiere la clave `region`. La región debe ser la región de AWS en la que se ejecuta el agente. El agente usa `boto3` para cargar las credenciales predeterminadas de AWS.

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <región-aws>
registry:
  type: ecr
  # URI del repositorio de ECR donde el agente almacenará imágenes.
  # Asegúrate de que la región coincida con lo que has configurado en tu
  # entorno.
  uri: <id-de-cuenta>.ecr.<región-aws>.amazonaws.com/<nombre-del-repositorio>
  # Alternativamente, puedes simplemente configurar el nombre del repositorio
  # repository: mi-nombre-de-repositorio
```

Consulta la [documentación de boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) para obtener más información sobre cómo configurar las credenciales predeterminadas de AWS.

  </TabItem>
  <TabItem value="gcp">

El entorno de GCP requiere claves de `region` y `project`. Establece la `region` a la región de GCP en la que se ejecuta el agente. Establece el `project` de GCP en el que se ejecuta el agente. El agente usa `google.auth.default()` para cargar las credenciales predeterminadas de GCP.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <región-gcp>
  project: <id-del-proyecto-gcp>
registry:
  # Requiere una configuración de entorno de gcp.
  type: gcr
  # URI del repositorio de Artifact Registry y nombre de la imagen donde el agente
  # almacenará imágenes. Asegúrate de que la región y el proyecto coincidan con lo que has
  # configurado en tu entorno.
  uri: <región>-docker.pkg.dev/<id-del-proyecto>/<nombre-del-repositorio>/<nombre-de-la-imagen>
  # Alternativamente, puedes configurar las claves de repositorio e image-name.
  # repository: mi-repositorio-de-artefactos
  # image-name: mi-nombre-de-imagen
```

Consulta la [documentación de `google-auth`](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) para obtener más información sobre cómo configurar las credenciales predeterminadas de GCP.

  </TabItem>
  <TabItem value="azure">

El entorno de Azure no requiere claves adicionales. Cuando el agente inicia, usa `azure.identity.DefaultAzureCredential()` para cargar las credenciales predeterminadas de Azure.

```yaml title="launch-config.yaml"
environment:
  type: azure
registry:
  type: acr
  uri: https://mi-registro.azurecr.io/mi-repositorio
```

Consulta la [documentación de `azure-identity`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) para obtener más información sobre cómo configurar las credenciales predeterminadas de Azure.

  </TabItem>
</Tabs>

## Permisos del agente

Los permisos requeridos por el agente dependen de tu caso de uso. Las políticas descritas a continuación son utilizadas por los agentes de lanzamiento.

### Permisos para el registro en la nube

A continuación, se muestran los permisos que generalmente son requeridos por los agentes de lanzamiento para interactuar con registros en la nube.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

```yaml
{
  'Version': '2012-10-17',
  'Statement':
    [
      {
        'Effect': 'Allow',
        'Action':
          [
            'ecr:CreateRepository',
            'ecr:UploadLayerPart',
            'ecr:PutImage',
            'ecr:CompleteLayerUpload',
            'ecr:InitiateLayerUpload',
            'ecr:DescribeRepositories',
            'ecr:DescribeImages',
            'ecr:BatchCheckLayerAvailability',
            'ecr:BatchDeleteImage',
          ],
        'Resource': 'arn:aws:ecr:<región>:<id-de-cuenta>:repositorio/<repositorio>',
      },
      {
        'Effect': 'Allow',
        'Action': 'ecr:GetAuthorizationToken',
        'Resource': '*',
      },
    ],
}
```

  </TabItem>
  <TabItem value="gcp">

```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

  </TabItem>
  <TabItem value="azure">

Añade el rol [`AcrPush`](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush) si usas el constructor Kaniko.

</TabItem>
</Tabs>

### Permisos de Kaniko

El agente de lanzamiento requiere permiso para empujar al almacenamiento en la nube si el agente usa el constructor Kaniko. Kaniko usa un almacén de contexto fuera del pod que ejecuta el trabajo de construcción.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

El almacén de contexto recomendado para el constructor Kaniko en AWS es Amazon S3. La siguiente política se puede usar para dar acceso al agente a un bucket de S3:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<NOMBRE-DEL-BUCKET>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<NOMBRE-DEL-BUCKET>/*"]
    }
  ]
}
```

  </TabItem>
  <TabItem value="gcp">

En GCP, se requieren los siguientes permisos de IAM para que el agente suba contextos de construcción a GCS:

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

  </TabItem>
  <TabItem value="azure">

El rol de [Contribuidor de Datos de Blob de Almacenamiento](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) es necesario para que el agente suba contextos de construcción a Azure Blob Storage.

  </TabItem>
</Tabs>

### Permisos para ejecutar trabajos

El agente necesita permiso en tu nube de AWS o GCP para iniciar trabajos en Amazon SageMaker o Vertex AI, respectivamente.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon SageMaker', value: 'aws'},
{label: 'Vertex AI', value: 'vertex'},
]}>
<TabItem value="aws">

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:CreateTrainingJob",
      "Resource": "arn:aws:sagemaker:<REGIÓN>:<ID-DE-CUENTA>/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "<ARN-DEL-ROL-A-PASAR>"
    },
    {
      "Effect": "Allow",
      "Action": "kms:CreateGrant",
      "Resource": "<ARN-DE-LLAVE-KMS>",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "sagemaker.<REGIÓN>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
```

:::note
El permiso `kms:CreateGrant` para colas de SageMaker es requerido solo si la Configuración de Recursos asociada tiene especificado un VolumeKmsKeyId y el rol asociado no tiene una política que permita esta acción.
:::

  </TabItem>
  <TabItem value="vertex">

Para ejecutar trabajos con Vertex AI también necesitas configurar un bucket de GCS y otorgar al agente los permisos descritos anteriormente.

```js
ml.jobs.create;
ml.jobs.list;
ml.jobs.get;
```

  </TabItem>
</Tabs>

### Credenciales de repositorio Git

Debes proporcionar credenciales a tu agente de lanzamiento si usas un repositorio git privado como fuente para tu trabajo de lanzamiento.

Las credenciales que especifiques dependen del tipo de repositorio que estés utilizando.

Los repositorios Git se acceden típicamente usando SSH o HTTPS. El tipo de URL determina qué protocolo se usa para acceder al repositorio. Consulta [Manejo de URL remotas de Git](./create-launch-job.md#git-remote-url-handling) en [Crear un trabajo de lanzamiento](./create-launch-job.md) para obtener más información sobre cómo crear trabajos que hagan referencia a un remoto de git con cualquiera de los protocolos.