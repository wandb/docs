---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vincular una versión de modelo

Vincula una versión de modelo a un modelo registrado con la aplicación W&B o programáticamente con el SDK de Python.

## Vincular un modelo programáticamente

Usa el método [`link_model`](../../ref/python/run.md#link_model) para registrar programáticamente archivos de modelo en un run de W&B y vincularlo al [Registro de Modelos de W&B](./intro.md).

Asegúrate de reemplazar los valores encerrados en `<>` por los tuyos:

```python
import wandb

run = wandb.init(entity="<entity>", project="<proyecto>")
run.link_model(path="<ruta-al-modelo>", registered_model_name="<nombre-modelo-registrado>")
run.finish()
```

W&B crea un modelo registrado para ti si el nombre que especificas para el parámetro `nombre-modelo-registrado` no existe ya.

Por ejemplo, supongamos que tienes un modelo registrado existente llamado "Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`) en tu Registro de Modelos. Y supongamos que algunas versiones de modelo están vinculadas a él: `v0`, `v1`, `v2`. Si vinculas programáticamente un nuevo modelo y usas el mismo nombre de modelo registrado (`registered-model-name="Fine-Tuned-Review-Autocompletion"`), W&B vincula este modelo al modelo registrado existente y le asigna una versión de modelo `v3`. Si no existe ningún modelo registrado con este nombre, se crea un nuevo modelo registrado y tendrá una versión de modelo `v0`.

Consulta un ejemplo de ["Fine-Tuned-Review-Autocompletion" modelo registrado aquí](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## Vincular un modelo interactivamente
Vincula un modelo interactivamente con el Registro de Modelos o con el navegador de Artefactos.

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Registro de Modelos', value: 'model_ui'},
    {label: 'Navegador de Artefactos', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. Navega a la aplicación de Registro de Modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Pasa el mouse al lado del nombre del modelo registrado al que quieres vincular un nuevo modelo.
3. Selecciona el icono del menú de albóndigas (tres puntos horizontales) al lado de **Ver detalles**.
4. Desde el menú desplegable, selecciona **Vincular nueva versión**.
5. Desde el menú desplegable **Proyecto**, selecciona el nombre del proyecto que contiene tu modelo.
6. Desde el menú desplegable **Artefacto de Modelo**, selecciona el nombre del artefacto de modelo.
7. Desde el menú desplegable **Versión**, selecciona la versión de modelo que quieres vincular al modelo registrado.

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. Navega al navegador de artefactos de tu proyecto en la aplicación W&B en: `https://wandb.ai/<entity>/<proyecto>/artifacts`
2. Selecciona el icono de Artefactos en la barra lateral izquierda.
3. Haz clic en la versión del modelo que quieres vincular a tu registro.
4. Dentro de la sección **Resumen de la versión**, haz clic en el botón **Vincular a registro**.
5. Desde el modal que aparece a la derecha de la pantalla, selecciona un modelo registrado desde el menú desplegable **Seleccionar un modelo registrado**.
6. Haz clic en **Siguiente paso**.
7. (Opcional) Selecciona un alias del menú desplegable **Alias**.
8. Haz clic en **Vincular a registro**.

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## Ver la fuente de los modelos vinculados

Hay dos formas de ver la fuente de los modelos vinculados: El navegador de artefactos dentro del proyecto al que se registró el modelo y el Registro de Modelos de W&B.

Un puntero conecta una versión de modelo específica en el registro de modelos a la fuente del artefacto de modelo (ubicada dentro del proyecto al que se registró el modelo). El artefacto de modelo fuente también tiene un puntero al registro de modelos.

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Registro de Modelos', value: 'registry'},
    {label: 'Navegador de Artefactos', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navega a tu registro de modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Selecciona **Ver detalles** al lado del nombre de tu modelo registrado.
3. Dentro de la sección **Versiones**, selecciona **Ver** al lado de la versión de modelo que quieres investigar.
4. Haz clic en la pestaña **Versión** dentro del panel derecho.
5. Dentro de la sección **Resumen de la versión** hay una fila que contiene un campo **Versión Fuente**. El campo **Versión Fuente** muestra tanto el nombre del modelo como la versión del modelo.

Por ejemplo, la siguiente imagen muestra una versión de modelo `v0` llamada `mnist_model` (ver campo **Versión Fuente** `mnist_model:v0`), vinculada a un modelo registrado llamado `MNIST-dev`.

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. Navega al navegador de artefactos de tu proyecto en la aplicación W&B en: `https://wandb.ai/<entity>/<proyecto>/artifacts`
2. Selecciona el icono de Artefactos en la barra lateral izquierda.
3. Expande el menú desplegable **modelo** desde el panel de Artefactos.
4. Selecciona el nombre y la versión del modelo vinculado al registro de modelos.
5. Haz clic en la pestaña **Versión** dentro del panel derecho.
6. Dentro de la sección **Resumen de la versión** hay una fila que contiene un campo **Vinculado A**. El campo **Vinculado A** muestra tanto el nombre del modelo registrado como la versión que posee(`nombre-modelo-registrado:versión`).

Por ejemplo, en la siguiente imagen, hay un modelo registrado llamado `MNIST-dev` (ver el campo **Vinculado A**). Una versión de modelo llamada `mnist_model` con una versión `v0`(`mnist_model:v0`) apunta al modelo registrado `MNIST-dev`.


![](/images/models/view_linked_model_artifacts_browser.png)


  </TabItem>
</Tabs>