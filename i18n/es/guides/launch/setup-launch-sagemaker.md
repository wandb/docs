---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Configuración para SageMaker

Puede utilizar W&B Launch para enviar trabajos de lanzamiento a Amazon SageMaker para entrenar modelos de aprendizaje automático utilizando algoritmos proporcionados o personalizados en la plataforma SageMaker. SageMaker se encarga de iniciar y liberar recursos de cómputo, por lo que puede ser una buena opción para equipos sin un cluster EKS.

Los trabajos de lanzamiento enviados a Amazon SageMaker se ejecutan como Trabajos de Entrenamiento de SageMaker con la [API CreateTrainingJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). Los argumentos para la API `CreateTrainingJob` se controlan con la configuración de la cola de lanzamiento.

Amazon SageMaker [utiliza imágenes Docker para ejecutar trabajos de entrenamiento](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html). Las imágenes que SageMaker extrae deben almacenarse en el Amazon Elastic Container Registry (ECR). Esto significa que la imagen que utilice para el entrenamiento debe estar almacenada en ECR. Para obtener más información sobre cómo configurar Launch con ECR, consulte [Configuración avanzada del agente](./setup-agent-advanced.md).

Amazon SageMaker requiere un rol de ejecución IAM. El rol IAM se utiliza dentro de las instancias de trabajos de entrenamiento de SageMaker para controlar el acceso a recursos requeridos como ECR y Amazon S3. Tome nota del ARN del rol IAM. Necesitará especificar el ARN del rol IAM en su configuración de cola.

## Prerrequisitos
Cree y tome nota de los siguientes recursos de AWS:

1. **Configurar SageMaker en su cuenta de AWS.** Consulte la [Guía del Desarrollador de SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) para más información.
2. **Crear un repositorio de Amazon ECR** para almacenar imágenes que desea ejecutar en Amazon SageMaker. Consulte la [documentación de Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) para más información.
3. **Crear buckets de Amazon S3** para almacenar las entradas y salidas de SageMaker para sus trabajos de entrenamiento de SageMaker. Consulte la [documentación de Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) para más información. Tome nota del URI del bucket de S3 y el directorio.
4. **Crear rol de ejecución IAM.** El rol utilizado en el trabajo de entrenamiento de SageMaker requiere los siguientes permisos para funcionar. Estos permisos permiten registrar eventos, extraer de ECR e interactuar con los buckets de entrada y salida.  (Nota: si ya tiene este rol para trabajos de entrenamiento de SageMaker, no necesita crearlo nuevamente.)
  ```json title="Política del rol IAM"
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:CreateLogGroup",
          "logs:DescribeLogStreams",
          "ecr:GetAuthorizationToken"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::<bucket-de-entrada>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject"
        ],
        "Resource": [
          "arn:aws:s3:::<bucket-de-entrada>/<objeto>",
          "arn:aws:s3:::<bucket-de-salida>/<ruta>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        "Resource": "arn:aws:ecr:<región>:<id-de-cuenta>:repository/<repo>"
      }
    ]
  }
  ```
  Tome nota del ARN del rol IAM. Proporcionará el ARN del rol creado en este paso cuando configure la cola de lanzamiento.
5. **Crear un rol IAM para el agente de lanzamiento** El agente de lanzamiento necesita permiso para crear trabajos de entrenamiento de SageMaker. Adjunte la siguiente política al rol IAM que utilizará para el agente de lanzamiento. Tome nota del ARN del rol IAM que cree para el agente de lanzamiento:

  ```yaml
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "sagemaker:AddTags",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:sagemaker:<región>:<id-de-cuenta>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<id-de-cuenta>:role/<RoleArn-de-config-de-cola>"
      },
      {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-DE-LLAVE-KMS>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "sagemaker.<región>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
  
  


:::note
* Si desea que el agente de lanzamiento construya imágenes, consulte la [Configuración avanzada del agente](./setup-agent-advanced.md) para obtener permisos adicionales requeridos.
* El permiso `kms:CreateGrant` para colas de SageMaker solo es necesario si la ResourceConfig asociada tiene un VolumeKmsKeyId especificado y el rol asociado no tiene una política que permita esta acción.
:::

## Configurar una cola para SageMaker
Cree una cola en la aplicación W&B que use SageMaker como su recurso de cómputo:

1. Navegue a la [Aplicación de Launch](https://wandb.ai/launch).
3. Haga clic en el botón **Crear Cola**.
4. Seleccione la **Entidad** en la que le gustaría crear la cola.
5. Proporcione un nombre para su cola en el campo **Nombre**.
6. Seleccione **SageMaker** como el **Recurso**.
7. Dentro del campo **Configuración**, proporcione información sobre su trabajo de SageMaker. Por defecto, W&B llenará un YAML y un cuerpo de solicitud de JSON `CreateTrainingJob`:
```json
{
  "RoleArn": "<REQUERIDO>",
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<REQUERIDO>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
Debe especificar al mínimo:

- `RoleArn` : ARN del rol IAM que creó y que cumplió con los [prerrequisitos](#prerequisites).
- `OutputDataConfig.S3OutputPath` : Un URI de Amazon S3 que especifica dónde se almacenarán las salidas de SageMaker.
- `ResourceConfig`: Especificación requerida de una configuración de recurso. Las opciones para la configuración de recursos se describen [aquí](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html).
- `StoppingCondition`: Especificación requerida de las condiciones de detención para el trabajo de entrenamiento. Las opciones se describen [aquí](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html).
7. Haga clic en el botón **Crear Cola**.

## Configurar un agente de lanzamiento
Configure el agente de lanzamiento con un archivo de configuración YAML llamado `launch-config.yaml`. Por defecto, W&B buscará el archivo de configuración en `~/.config/wandb/launch-config.yaml`. Puede especificar opcionalmente un directorio diferente cuando active el agente de lanzamiento.

El siguiente fragmento YAML demuestra cómo especificar las opciones de configuración principal del agente:

```yaml title="launch-config.yaml"
max_jobs: <n-trabajos-concurrentes>
queues:
  - <nombre-de-la-cola>
```

:::tip
Hay dos maneras de enviar lanzamientos en Amazon SageMaker:
* Opción 1: Traiga su propia imagen (BYOI) y súbala a su repositorio de Amazon ECR.
* Opción 2: Permita que el agente de lanzamiento de W&B construya un contenedor para usted y lo suba a su repositorio de ECR.

Necesitará proporcionar información adicional a la configuración de su agente de lanzamiento si desea que el agente construya imágenes para usted (Opción 2). Para más información, consulte [Configuración avanzada del agente](./setup-agent-advanced.md).
:::

## Configurar permisos del agente para Amazon SageMaker
Los roles IAM se pueden asociar con agentes de lanzamiento de varias maneras. Cómo configure estos roles dependerá en parte de dónde esté sondeando su agente de lanzamiento.


Dependiendo de su caso de uso, consulte la siguiente guía.

### Agente sondea desde una máquina local 

Use los archivos de configuración de AWS ubicados en `~/.aws/config` y `~/.aws/credentials` para asociar un rol con un agente que esté sondeando en una máquina local. Proporcione el ARN del rol IAM que creó para el agente de lanzamiento en el paso anterior.
 
```yaml title="~/.aws/config"
[profile sagemaker-agent]
role_arn = arn:aws:iam::<id-de-cuenta>:role/<nombre-del-rol-del-agente>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<id-de-clave-de-acceso>
aws_secret_access_key=<clave-de-acceso-secreta>
aws_session_token=<token-de-sesión>
```

### Agente sondea dentro de AWS (como EC2)
Puede usar un rol de instancia para proporcionar permisos al agente si desea ejecutar el agente dentro de un servicio de AWS como EC2.