---
description: W&B's Embedding Projector allows users to plot multi-dimensional embeddings
  on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE.
displayed_sidebar: default
---

# Proyector de Embeddings

![](/images/weave/embedding_projector.png)

[Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) se utilizan para representar objetos (personas, imágenes, publicaciones, palabras, etc.) con una lista de números, a veces denominada _vector_. En casos de uso de ciencia de datos y aprendizaje automático, los embeddings pueden generarse utilizando una variedad de enfoques a través de una gama de aplicaciones. Esta página asume que el lector está familiarizado con los embeddings y está interesado en analizarlos visualmente dentro de W&B.

## Ejemplos de Embeddings

Puedes saltar directamente a un [Reporte de Demo Interactivo en Vivo](https://wandb.ai/timssweeney/toy\_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) o ejecutar el código de este reporte desde el [Colab de Ejemplo](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm\_).

### Hola mundo

W&B te permite registrar embeddings utilizando la clase `wandb.Table`. Considera el siguiente ejemplo de 3 embeddings, cada uno consistiendo en 5 dimensiones:

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # embedding 1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # embedding 2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # embedding 3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

Después de ejecutar el código anterior, el panel de control de W&B tendrá una nueva Tabla que contiene tus datos. Puedes seleccionar `Proyección 2D` desde el selector del panel superior derecho para trazar los embeddings en 2 dimensiones. Los valores predeterminados inteligentes serán seleccionados automáticamente, lo cual puede ser fácilmente reemplazado en el menú de configuración accediendo a través del icono de engranaje. En este ejemplo, automáticamente utilizamos todas las 5 dimensiones numéricas disponibles.

![](/images/app_ui/weave_hello_world.png)

### Dígitos MNIST

Mientras que el ejemplo anterior muestra la mecánica básica de registrar embeddings, típicamente estás trabajando con muchas más dimensiones y muestras. Consideremos el dataset de Dígitos MNIST ([dataset de dígitos escritos a mano del UCI ML](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)) disponible a través de [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_digits.html). Este dataset tiene 1797 registros, cada uno con 64 dimensiones. El problema es un caso de uso de clasificación de 10 clases. También podemos convertir los datos de entrada en una imagen para visualización.

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# Cargar el dataset
ds = load_digits(as_frame=True)
df = ds.data

# Crear una columna "target"
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# Crear una columna "image"
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

Después de ejecutar el código anterior, nuevamente nos presentan una Tabla en la UI. Al seleccionar `Proyección 2D` podemos configurar la definición del embedding, colorear, algoritmo (PCA, UMAP, t-SNE), parámetros del algoritmo e incluso superposición (en este caso mostramos la imagen al pasar el cursor sobre un punto). En este caso particular, estos son todos "valores predeterminados inteligentes" y deberías ver algo muy similar con un solo clic en `Proyección 2D`. ([Haz clic aquí para interactuar](https://wandb.ai/timssweeney/embedding\_tutorial/runs/k6guxhum?workspace=user-timssweeney) con este ejemplo).

![](/images/weave/embedding_projector.png)

## Opciones de Registro

Puedes registrar embeddings en una serie de formatos diferentes:

1. **Columna de Embedding Única:** A menudo tus datos ya están en formato "matriz". En este caso, puedes crear una única columna de embedding - donde el tipo de datos de los valores de las celdas puede ser `list[int]`, `list[float]`, o `np.ndarray`.
2. **Múltiples Columnas Numéricas:** En los dos ejemplos anteriores, usamos este enfoque y creamos una columna para cada dimensión. Actualmente aceptamos `int` o `float` de python para las celdas.

![Columna de Embedding Única](/images/weave/logging_options.png)
![Muchas Columnas Numéricas](/images/weave/logging_option_image_right.png)

Además, al igual que todas las tablas, tienes muchas opciones en cuanto a cómo construir la tabla:

1. Directamente de un **dataframe** usando `wandb.Table(dataframe=df)`
2. Directamente de una **lista de datos** usando `wandb.Table(data=[...], columns=[...])`
3. Construir la tabla **incrementalmente fila por fila** (genial si tienes un bucle en tu código). Añade filas a tu tabla usando `table.add_data(...)`
4. Añadir una **columna de embedding** a tu tabla (genial si tienes una lista de predicciones en forma de embeddings): `table.add_col("col_name", ...)`
5. Añadir una **columna calculada** (genial si tienes una función o modelo que quieres mapear sobre tu tabla): `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## Opciones de Trazado

Después de seleccionar `Proyección 2D`, puedes hacer clic en el icono de engranaje para editar la configuración de renderizado. Además de seleccionar las columnas previstas (ver arriba), puedes seleccionar un algoritmo de interés (junto con los parámetros deseados). A continuación, puedes ver los parámetros para UMAP y t-SNE respectivamente.

![](/images/weave/plotting_options_left.png) 
![](/images/weave/plotting_options_right.png)

:::info
Nota: actualmente reducimos la muestra a un subconjunto aleatorio de 1000 filas y 50 dimensiones para los tres algoritmos.
:::