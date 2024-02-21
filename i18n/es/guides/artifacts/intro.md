---
description: Overview of what W&B Artifacts are, how they work, and how to get started
  using W&B Artifacts.
slug: /guides/artifacts
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Artefactos

<CTAButtons productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb"/>

Usa los Artefactos de W&B para hacer seguimiento y versionar cualquier dato serializado como las entradas y salidas de tus [W&B Runs](../runs/intro.md). Por ejemplo, un run de entrenamiento del modelo podría tomar un dataset como entrada y un modelo entrenado como salida. Además de registrar hiperparámetros y metadatos en un run, puedes usar un artefacto para registrar el dataset usado para entrenar el modelo como entrada y los checkpoints del modelo resultante como salidas. Siempre podrás responder a la pregunta "¿en qué versión de mi dataset fue entrenado este modelo?".

En resumen, con los Artefactos de W&B, puedes:
* [Ver de dónde proviene un modelo, incluyendo los datos en los que fue entrenado](./explore-and-traverse-an-artifact-graph.md).
* [Versionar cada cambio de dataset o checkpoint del modelo](./create-a-new-artifact-version.md).
* [Reutilizar fácilmente modelos y datasets a través de tu equipo](./download-and-use-an-artifact.md).

![](/images/artifacts/artifacts_landing_page2.png)

El diagrama anterior demuestra cómo puedes usar artefactos a lo largo de todo tu flujo de trabajo de ML; como entradas y salidas de [runs](../runs/intro.md).

## Cómo funciona

Crea un artefacto con cuatro líneas de código:
1. Crea un [W&B run](../runs/intro.md).
2. Crea un objeto artefacto con la API [`wandb.Artifact`](../../ref/python/artifact.md).
3. Añade uno o más archivos, como un archivo de modelo o dataset, a tu objeto artefacto.
4. Registra tu artefacto en W&B.


```python showLineNumbers
run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="my_data", type="dataset")
artifact.add_dir(local_path="./dataset.h5")  # Añade directorio de dataset al artefacto
run.log_artifact(artifact)  # Registra la versión del artefacto "my_data:v0"
```

:::tip
El fragmento de código anterior, y el [Colab Notebook vinculado en esta página](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb), muestran cómo hacer seguimiento de archivos subiéndolos a W&B. Consulta la página de [seguimiento de archivos externos](./track-external-files.md) para obtener información sobre cómo añadir referencias a archivos o directorios almacenados en almacenamiento de objetos externos (por ejemplo, en un bucket de Amazon S3).
:::

## Cómo empezar

Dependiendo de tu caso de uso, explora los siguientes recursos para comenzar con los Artefactos de W&B:

* Si es la primera vez que usas Artefactos de W&B, te recomendamos que revises el [Colab Notebook de Artefactos](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb#scrollTo=fti9TCdjOfHT).
* Lee el [paseo por los artefactos](./artifacts-walkthrough.md) para obtener un esquema paso a paso de los comandos del SDK de Python de W&B que podrías usar para crear, hacer seguimiento y usar un artefacto de dataset.
* Explora este capítulo para aprender cómo:
  * [Construir un artefacto](./construct-an-artifact.md) o una [nueva versión de artefacto](./create-a-new-artifact-version.md)
  * [Actualizar un artefacto](./update-an-artifact.md)
  * [Descargar y usar un artefacto](./download-and-use-an-artifact.md).
  * [Eliminar artefactos](./delete-artifacts.md).
* Explora las [APIs de Artefacto del SDK de Python](../../ref/python/artifact.md) y la [Guía de Referencia de CLI de Artefacto](../../ref/cli/wandb-artifact/README.md).