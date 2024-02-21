---
displayed_sidebar: default
---

# Guardado de código

Por defecto, solo guardamos el último hash de commit de git. Puedes activar más funciones de código para comparar el código entre tus experimentos dinámicamente en la UI.

A partir de la versión 0.8.28 de `wandb`, podemos guardar el código de tu archivo principal de entrenamiento donde llamas a `wandb.init()`. Esto se sincronizará con el panel de control y aparecerá en una pestaña en la página del run, así como en el panel Comparador de Código. Ve a tu [página de configuración](https://app.wandb.ai/settings) para habilitar el guardado de código por defecto.

![Así es como se ven los ajustes de tu cuenta. Puedes guardar código por defecto.](/images/app_ui/code_saving.png)

## Guardar Código de Biblioteca

Cuando el guardado de código está habilitado, wandb guardará el código del archivo que llamó a `wandb.init()`. Para guardar código adicional de la biblioteca, tienes dos opciones:

* Llamar a `wandb.run.log_code(".")` después de llamar a `wandb.init()`
* Pasar un objeto de configuración a `wandb.init` con code\_dir establecido: `wandb.init(settings=wandb.Settings(code_dir="."))`

Esto capturará todos los archivos de código fuente de Python en el directorio actual y todos los subdirectorios como un [artefacto](../../../../ref/python/artifact.md). Para tener más control sobre los tipos y ubicaciones de los archivos de código fuente que se guardan, consulta los [documentos de referencia](../../../../ref/python/run.md#log_code).

## Comparador de Código

Haz clic en el botón **+** en tu espacio de trabajo o reporte para agregar un nuevo panel, y selecciona el Comparador de Código. Diferencia cualquier par de experimentos en tu proyecto y ve exactamente qué líneas de código cambiaron. Aquí tienes un ejemplo:

![](/images/app_ui/code_comparer.png)

## Historial de Sesión de Jupyter

A partir de la versión 0.8.34 de **wandb**, nuestra biblioteca realiza el guardado de sesiones de Jupyter. Cuando llamas a **wandb.init()** dentro de Jupyter, añadimos un hook para guardar automáticamente un notebook de Jupyter que contiene el historial de código ejecutado en tu sesión actual. Puedes encontrar este historial de sesión en un navegador de archivos de runs bajo el directorio de código:

![](/images/app_ui/jupyter_session_history.png)

Hacer clic en este archivo mostrará las celdas que se ejecutaron en tu sesión junto con cualquier salida creada al llamar al método de visualización de iPython. Esto te permite ver exactamente qué código se ejecutó dentro de Jupyter en un run dado. Cuando es posible, también guardamos la versión más reciente del notebook que encontrarías en el directorio de código también.

![](/images/app_ui/jupyter_session_history_display.png)

## Diferencias en Jupyter

Una última característica adicional es la capacidad de diferenciar notebooks. En lugar de mostrar el JSON en bruto en nuestro panel Comparador de Código, extraemos cada celda y mostramos cualquier línea que haya cambiado. Tenemos algunas características emocionantes planeadas para integrar Jupyter más profundamente en nuestra plataforma.