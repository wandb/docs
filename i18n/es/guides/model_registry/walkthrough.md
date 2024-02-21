---
description: Learn how to use W&B for Model Management
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Recorrido

El siguiente recorrido muestra cómo registrar un modelo en W&B. Al final del recorrido, usted:

* Creará y entrenará un modelo con el dataset MNIST y el framework de Keras.
* Registrará el modelo que entrenó en un proyecto de W&B
* Marcará el dataset utilizado como una dependencia del modelo que creó
* Vinculará el modelo al Registro de Modelos de W&B.
* Evaluará el rendimiento del modelo que vinculó al registro
* Marcará una versión del modelo lista para producción.

:::note
* Copie los fragmentos de código en el orden presentado en esta guía.
* El código que no es único para el Registro de Modelos se oculta en celdas colapsables.
:::

## Configuración

Antes de comenzar, importe las dependencias de Python requeridas para este recorrido:

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
```

Proporcione su entidad de W&B a la variable `entity`: 

```python
entity = "<entidad>"
```

### Crear un artefacto de dataset

Primero, cree un dataset. El siguiente fragmento de código crea una función que descarga el dataset MNIST:
```python
def generate_raw_data(train_size=6000):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    print("Generados {} filas de datos de entrenamiento.".format(train_size))
    print("Generados {} filas de datos de evaluación.".format(eval_size))

    return (x_train[:train_size], y_train[:train_size]), (
        x_eval[:eval_size],
        y_eval[:eval_size],
    )

# Crear dataset
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

A continuación, suba el dataset a W&B. Para hacer esto, cree un objeto [artefacto](../artifacts/intro.md) y agregue el dataset a ese artefacto. 

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# Iniciar un run de W&B
run = wandb.init(entity=entity, project=project, job_type=job_type)

# Crear tabla W&B para datos de entrenamiento
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# Crear tabla W&B para datos de evaluación
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# Crear un objeto artefacto
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# Agregar objeto wandb.WBValue al artefacto.
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# Persistir cualquier cambio hecho al artefacto.
artifact.save()

# Indicar a W&B que este run ha finalizado.
run.finish()
```

:::tip
Almacenar archivos (como datasets) en un artefacto es útil en el contexto de registrar modelos porque le permite rastrear las dependencias de un modelo.
:::

## Entrenar un modelo
Entrene un modelo con el dataset de artefacto que creó en el paso anterior.

### Declarar el artefacto de dataset como una entrada al run

Declare el artefacto de dataset que creó en un paso anterior como la entrada al run de W&B. Esto es particularmente útil en el contexto de registrar modelos porque declarar un artefacto como entrada a un run le permite rastrear el dataset (y la versión del dataset) utilizado para entrenar un modelo específico. W&B utiliza la información recopilada para crear un [mapa de linaje](./model-lineage.md). 

Use la API `use_artifact` para declarar el artefacto de dataset como la entrada del run y para recuperar el propio artefacto. 

```python
job_type = "train_model"
config = {
    "optimizador": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

# Iniciar un run de W&B
run = wandb.init(project=project, job_type=job_type, config=config)

# Recuperar el artefacto de dataset
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# Obtener contenido específico del dataframe
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

Para obtener más información sobre el seguimiento de las entradas y salida de un modelo, consulte [Crear linaje del modelo](./model-lineage.md) mapa.

### Definir y entrenar modelo

Para este recorrido, defina una Red Neuronal Convolucional (CNN) 2D con Keras para clasificar imágenes del dataset MNIST. 

<details>
<summary>Entrenar CNN en datos MNIST</summary>

```python
# Almacenar valores de nuestro diccionario de configuración en variables para un fácil acceso
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizador"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# Crear arquitectura del modelo
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Generar etiquetas para datos de entrenamiento
y_train = keras.utils.to_categorical(y_train, num_classes)

# Crear conjunto de entrenamiento y prueba
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
```
A continuación, entrene el modelo:

```python
# Entrenar el modelo
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

Finalmente, guarde el modelo localmente en su máquina: 

```python
# Guardar modelo localmente
path = "model.h5"
model.save(path)
```
</details>

## Registrar y vincular un modelo al Registro de Modelos
Use la API [`link_model`](../../ref/python/run.md#link_model) para registrar uno o más archivos de modelo en un run de W&B y vincularlo al [Registro de Modelos de W&B](./intro.md).

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

W&B crea un modelo registrado para usted si el nombre que especifica para `registered-model-name` no existe ya. 

Consulte [`link_model`](../../ref/python/run.md#link_model) en la guía de Referencia de la API para más información sobre parámetros opcionales.

## Evaluar el rendimiento de un modelo
Es una práctica común evaluar el rendimiento de uno o más modelos. 

Primero, obtenga el artefacto de dataset de evaluación almacenado en W&B en un paso anterior.

```python
job_type = "evaluate_model"

# Iniciar un run
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# Obtener artefacto de dataset, marcarlo como una dependencia
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# Obtener dataframe deseado
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

Descargue la [versión del modelo](./model-management-concepts.md#model-version) de W&B que desea evaluar. Use la API `use_model` para acceder y descargar su modelo.

```python
alias = "latest"  # alias
name = "mnist_model"  # nombre del artefacto del modelo

# Acceder y descargar modelo. Devuelve la ruta al artefacto descargado
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Cargue el modelo de Keras y calcule la pérdida:

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

Finalmente, registre la métrica de pérdida en el run de W&B:

```python
# # Registrar métricas, imágenes, tablas o cualquier dato útil para la evaluación.
run.log(data={"loss": (loss, _)})
```

## Promocionar una versión del modelo 
Marque una versión del modelo lista para la siguiente etapa de su flujo de trabajo de aprendizaje automático con un [*alias del modelo*](./model-management-concepts.md#model-alias). Cada modelo registrado puede tener uno o más alias de modelo. Un alias de modelo solo puede pertenecer a una única versión del modelo a la vez.

Por ejemplo, suponga que después de evaluar el rendimiento de un modelo, está seguro de que el modelo está listo para producción. Para promocionar esa versión del modelo, agregue el alias `production` a esa versión específica del modelo. 

:::tip
El alias `production` es uno de los alias más comunes utilizados para marcar un modelo como listo para producción.
:::

Puede agregar un alias a una versión del modelo de forma interactiva con la UI de la App de W&B o programáticamente con el SDK de Python. Los siguientes pasos muestran cómo agregar un alias con la App de Registro de Modelos de W&B:


1. Navegue a la App de Registro de Modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Haga clic en **Ver detalles** al lado del nombre de su modelo registrado.
3. Dentro de la sección **Versiones**, haga clic en el botón **Ver** al lado del nombre de la versión del modelo que desea promocionar. 
4. Al lado del campo **Aliases**, haga clic en el icono de más (**+**). 
5. Escriba `production` en el campo que aparece.
6. Presione Enter en su teclado.


![](/images/models/promote_model_production.gif)