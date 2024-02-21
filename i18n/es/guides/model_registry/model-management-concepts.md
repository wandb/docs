---
description: Model Registry terms and concepts
displayed_sidebar: default
---

# Términos y conceptos

<head>
  <title>Términos y conceptos del registro de modelos</title>
</head>

Los siguientes términos describen componentes clave del Registro de Modelos de W&B: [*versión del modelo*](#model-version), [*artefacto del modelo*](#model-artifact), y [*modelo registrado*](#registered-model).

## Versión del modelo
Una versión del modelo representa un único checkpoint del modelo. Las versiones del modelo son una instantánea en un punto en el tiempo de un modelo y sus archivos dentro de un experimento.

Una versión del modelo es un directorio inmutable de datos y metadatos que describe un modelo entrenado. W&B sugiere que añadas archivos a tu versión del modelo que te permitan almacenar (y restaurar) la arquitectura de tu modelo y los parámetros aprendidos en una fecha posterior.

Una versión del modelo pertenece a uno, y solo uno, [artefacto del modelo](#model-artifact). Una versión del modelo puede pertenecer a cero o más, [modelos registrados](#registered-model). Las versiones del modelo se almacenan en un artefacto del modelo en el orden en que se registran al artefacto del modelo. W&B automáticamente crea una nueva versión del modelo si detecta que un modelo que registras (al mismo artefacto del modelo) tiene contenidos diferentes que una versión del modelo anterior.

Almacena archivos dentro de las versiones del modelo que se producen a partir del proceso de serialización proporcionado por tu biblioteca de modelado (por ejemplo, [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) y [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)).

<!-- [INSERTAR IMAGEN] -->

## Alias del modelo

Los alias del modelo son cadenas mutables que te permiten identificar o referenciar de manera única una versión del modelo en tu modelo registrado con un identificador semánticamente relacionado. Solo puedes asignar un alias a una versión de un modelo registrado. Esto es porque un alias debe referirse a una versión única cuando se usa programáticamente. También permite que los alias se usen para capturar el estado de un modelo (campeón, candidato, producción).

Es una práctica común usar alias como "mejor", "último", "producción" o "preparación" para marcar versiones del modelo con propósitos especiales.

Por ejemplo, supón que creas un modelo y le asignas un alias `"mejor"`. Puedes referirte a ese modelo específico con `run.use_model`

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Etiquetas del modelo
Las etiquetas del modelo son palabras clave o etiquetas que pertenecen a uno o más modelos registrados.

Usa etiquetas del modelo para organizar modelos registrados en categorías y para buscar sobre esas categorías en la barra de búsqueda del Registro de Modelos. Las etiquetas del modelo aparecen en la parte superior de la Tarjeta de Modelo Registrado. Podrías elegir usarlas para agrupar tus modelos registrados por tarea de ML, equipo propietario o prioridad. La misma etiqueta del modelo se puede agregar a múltiples modelos registrados para permitir la agrupación.

:::info
Las etiquetas del modelo, que son etiquetas aplicadas a modelos registrados para agrupación y descubrimiento, son diferentes de [alias del modelo](#model-alias). Los alias del modelo son identificadores únicos o apodos que usas para buscar una versión del modelo programáticamente. Para aprender más sobre cómo usar etiquetas para organizar las tareas en tu Registro de Modelos, consulta [Organizar modelos](./organize-models.md).
:::

## Artefacto del modelo
Un artefacto del modelo es una colección de [versiones del modelo](#model-version) registradas. Las versiones del modelo se almacenan en un artefacto del modelo en el orden en que se registran al artefacto del modelo.

Un artefacto del modelo puede contener una o más versiones del modelo. Un artefacto del modelo puede estar vacío si no se registran versiones del modelo en él.

Por ejemplo, supongamos que creas un artefacto del modelo. Durante el entrenamiento del modelo, guardas periódicamente tu modelo durante los checkpoints. Cada checkpoint corresponde a su propia [versión del modelo](#model-version). Todas las versiones del modelo creadas durante tu entrenamiento del modelo y el guardado de checkpoints se almacenan en el mismo artefacto del modelo que creaste al principio de tu script de entrenamiento.

<!-- y se le asignará un número de versión dependiendo de la secuencia en que fueron registrados. Una nueva versión se crea automáticamente cuando el contenido de la última versión que se registró ha cambiado.  -->


La imagen siguiente muestra un artefacto del modelo que contiene tres versiones del modelo: v0, v1 y v2.

![](@site/static/images/models/mr1c.png)

Ve un [ejemplo de artefacto del modelo aquí](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n).

## Modelo registrado
Un modelo registrado es una colección de punteros (enlaces) a versiones del modelo. Puedes pensar en un modelo registrado como una carpeta de "marcadores" de modelos candidatos para la misma tarea de ML. Cada "marcador" de un modelo registrado es un puntero a una [versión del modelo](#model-version) que pertenece a un [artefacto del modelo](#model-artifact). Puedes usar [etiquetas del modelo](#model-tags) para agrupar tus modelos registrados.

Los modelos registrados a menudo representan modelos candidatos para un único caso de uso o tarea de modelado. Por ejemplo, podrías crear un modelo registrado para diferentes tareas de clasificación de imágenes basado en el modelo que usas: "ImageClassifier-ResNet50", "ImageClassifier-VGG16", "DogBreedClassifier-MobileNetV2" y así sucesivamente. Las versiones del modelo se asignan números de versión en el orden en que se vincularon al modelo registrado.


Ve un [ejemplo de Modelo Registrado aquí](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions).