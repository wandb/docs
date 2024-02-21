---
description: Learn where W&B files are stored by default. Explore how to save, store
  sensitive information.
displayed_sidebar: default
---

# Privacidad de Datos y Cumplimiento

<head>
    <title>Privacidad de Datos y Cumplimiento de los Artefactos</title>
</head>

Los archivos se suben a un bucket de Google Cloud gestionado por W&B cuando registras artefactos. Los contenidos del bucket están cifrados tanto en reposo como en tránsito. Los archivos de artefactos solo son visibles para los usuarios que tienen acceso al proyecto correspondiente.

![Diagrama del Cliente y Servidor de GCS W&B](/images/artifacts/data_and_privacy_compliance_1.png)

Cuando eliminas una versión de un artefacto, se marca para eliminación suave en nuestra base de datos y se elimina de tu costo de almacenamiento. Cuando eliminas un artefacto completo, se pone en cola para su eliminación permanente y todos sus contenidos se eliminan del bucket de W&B. Si tienes necesidades específicas sobre la eliminación de archivos, por favor contacta a [Soporte al Cliente](mailto:support@wandb.com).

Para datasets sensibles que no pueden residir en un entorno multitenant, puedes usar un servidor W&B privado conectado a tu bucket en la nube o _artefactos de referencia_. Los artefactos de referencia rastrean referencias a buckets privados sin enviar contenidos de archivos a W&B. Los artefactos de referencia mantienen enlaces a archivos en tus buckets o servidores. En otras palabras, W&B solo mantiene seguimiento de los metadatos asociados con los archivos y no los archivos en sí.

![Diagrama del Cliente, Servidor y Nube de W&B](/images/artifacts/data_and_privacy_compliance_2.png)

Crea un artefacto de referencia similar a cómo creas un artefacto que no es de referencia:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

Para alternativas, contáctanos en [contact@wandb.com](mailto:contact@wandb.com) para hablar sobre instalaciones en la nube privada y on-prem.