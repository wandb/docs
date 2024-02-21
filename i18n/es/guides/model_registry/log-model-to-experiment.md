---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Seguimiento de un modelo

Realiza el seguimiento de un modelo, las dependencias del modelo y otra información relevante sobre ese modelo con el SDK de Python de W&B.

Bajo el capó, W&B crea un linaje de [artefacto del modelo](./model-management-concepts.md#model-artifact) que puedes visualizar con la interfaz de usuario de la aplicación W&B o programáticamente con el SDK de Python de W&B. Consulta el [Crear mapa del linaje del modelo](./model-lineage.md) para más información.

## Cómo registrar un modelo

Utiliza la API `run.log_model` para registrar un modelo. Proporciona la ruta donde se guardan los archivos de tu modelo al parámetro `path`. La ruta puede ser un archivo local, directorio o [URI de referencia](../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references) a un bucket externo como `s3://bucket/path`.

Opcionalmente proporciona un nombre para el artefacto del modelo para el parámetro `name`. Si `name` no se especifica, W&B utiliza el nombre base de la ruta de entrada precedido por el ID del run.

Copia y pega el siguiente fragmento de código. Asegúrate de reemplazar los valores encerrados en `<>` por los tuyos propios.

```python
import wandb

# Inicializar un run de W&B
run = wandb.init(project="<proyecto>", entity="<entidad>")

# Registrar el modelo
run.log_model(path="<ruta-al-modelo>", name="<nombre>")
```

<details>

<summary>Ejemplo: Registrar un modelo Keras en W&B</summary>

El siguiente ejemplo de código muestra cómo registrar un modelo de red neuronal convolucional (CNN) en W&B.

```python showLineNumbers
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Inicializar un run de W&B
run = wandb.init(entity="charlie", project="proyecto-mnist", config=config)

# Algoritmo de entrenamiento
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

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

# Guardar modelo
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# Registrar el modelo
# highlight-next-line
run.log_model(path=full_path, name="MNIST")

# Indicar explícitamente a W&B que finalice el run.
run.finish()
```
</details>