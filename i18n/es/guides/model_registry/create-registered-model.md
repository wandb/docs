---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Crear un modelo registrado

Cree un [modelo registrado](./model-management-concepts.md#registered-model) para contener todos los modelos candidatos para sus tareas de modelado. Puede crear un modelo registrado de forma interactiva dentro del Registro de Modelos o programáticamente con el SDK de Python.

## Crear un modelo registrado programáticamente
Registre un modelo programáticamente con el SDK de Python de W&B. W&B crea automáticamente un modelo registrado para usted si el modelo registrado no existe.

Asegúrese de reemplazar los valores encerrados en `<>` por los suyos:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

El nombre que proporcione para `registered_model_name` es el nombre que aparecerá en la [Aplicación de Registro de Modelos](https://wandb.ai/registry/model).

## Crear un modelo registrado de forma interactiva
Cree un modelo registrado de forma interactiva dentro de la [Aplicación de Registro de Modelos](https://wandb.ai/registry/model).

1. Navegue a la Aplicación de Registro de Modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Haga clic en el botón **Nuevo modelo registrado** ubicado en la parte superior derecha de la página del Registro de Modelos.
![](/images/models/create_registered_model_model_reg_app.png)
3. Del panel que aparece, seleccione la entidad a la que desea que pertenezca el modelo registrado desde el desplegable **Entidad Propietaria**.
![](/images/models/create_registered_model_3.png)
4. Proporcione un nombre para su modelo en el campo **Nombre**.
5. Desde el desplegable **Tipo**, seleccione el tipo de artefactos para vincular al modelo registrado.
6. (Opcional) Añada una descripción sobre su modelo en el campo **Descripción**.
7. (Opcional) Dentro del campo **Etiquetas**, añada una o más etiquetas.
8. Haga clic en **Registrar modelo**.


:::tip
Vincular manualmente un modelo al registro de modelos es útil para modelos únicos. Sin embargo, a menudo es útil [vincular programáticamente versiones de modelos al registro de modelos](#programmatically-link-a-model).

Por ejemplo, suponga que tiene un trabajo nocturno. Es tedioso vincular manualmente un modelo creado cada noche. En su lugar, podría crear un script que evalúe el modelo, y si el modelo mejora en rendimiento, vincule ese modelo al registro de modelos con el SDK de Python de W&B.
:::