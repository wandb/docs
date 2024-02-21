---
description: Create custom aliases for W&B Artifacts.
displayed_sidebar: default
---

# Crear un alias personalizado

<head>
    <title>Crea un alias personalizado para tu Artefacto.</title>
</head>

Utiliza alias como punteros a versiones específicas. Por defecto, `Run.log_artifact` añade el alias `latest` a la versión registrada.

Una versión del artefacto `v0` es creada y adjuntada a tu artefacto cuando registras un artefacto por primera vez. W&B hace un checksum de los contenidos cuando registras nuevamente al mismo artefacto. Si el artefacto cambió, W&B guarda una nueva versión `v1`.

Por ejemplo, si quieres que tu script de entrenamiento obtenga la versión más reciente de un dataset, especifica `latest` cuando uses ese artefacto. El siguiente ejemplo de código demuestra cómo descargar un artefacto de dataset reciente llamado `bike-dataset` que tiene un alias, `latest`:

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

También puedes aplicar un alias personalizado a una versión de artefacto. Por ejemplo, si quieres marcar que un checkpoint del modelo es el mejor en la métrica AP-50, podrías añadir la cadena `'best-ap50'` como un alias cuando registres el artefacto del modelo.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```