---
displayed_sidebar: default
---

# Crear mapa de linaje del modelo
Una característica útil de registrar artefactos de modelos en W&B son los gráficos de linaje. Los gráficos de linaje muestran los artefactos registrados por un run así como los artefactos utilizados por un run específico.

Esto significa que, cuando registras un artefacto de modelo, como mínimo tienes acceso a ver el run de W&B que utilizó o produjo el artefacto de modelo. Si [haces seguimiento de una dependencia](#track-an-artifact-dependency), también puedes ver las entradas utilizadas por el artefacto de modelo.

Por ejemplo, la imagen siguiente muestra artefactos creados y utilizados a lo largo de un experimento de ML:

![](/images/models/model_lineage_example.png)

De izquierda a derecha, la imagen muestra:
1. El run de W&B `jumping-monkey-1` creó el artefacto de dataset `mnist_dataset:v0`.
2. El run de W&B `vague-morning-5` entrenó un modelo utilizando el artefacto de dataset `mnist_dataset:v0`. La salida de este run de W&B fue un artefacto de modelo llamado `mnist_model:v0`.
3. Un run llamado `serene-haze-6` utilizó el artefacto de modelo (`mnist_model:v0`) para evaluar el modelo.

## Hacer seguimiento de una dependencia de artefacto

Declara un artefacto de dataset como una entrada a un run de W&B con la API `use_artifact` para hacer seguimiento de una dependencia.

El fragmento de código siguiente muestra cómo usar la API `use_artifact`:

```python
# Inicializar un run
run = wandb.init(project=project, entity=entity)

# Obtener artefacto, marcarlo como una dependencia
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

Una vez que hayas recuperado tu artefacto, puedes usar ese artefacto para, por ejemplo, evaluar el rendimiento de un modelo.

<details>

<summary>Ejemplo: Entrenar un modelo y hacer seguimiento de un dataset como la entrada de un modelo</summary>

```python
job_type = "train_model"

config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

# highlight-start
artifact = run.use_artifact(name)
# highlight-end

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# Almacenar valores de nuestro diccionario de configuración en variables para fácil acceso
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
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

# Generar etiquetas para los datos de entrenamiento
y_train = keras.utils.to_categorical(y_train, num_classes)

# Crear conjunto de entrenamiento y prueba
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# Entrenar el modelo
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# Guardar modelo localmente
path = "model.h5"
model.save(path)

path = "./model.h5"
registered_model_name = "MNIST-dev"
name = "mnist_model"

# highlight-start
run.link_model(path=path, registered_model_name=registered_model_name, name=name)
# highlight-end
run.finish()
```

</details>