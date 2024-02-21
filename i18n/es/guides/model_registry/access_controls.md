---
description: Use model registry role based access controls (RBAC) to control who can
  update protected aliases.
displayed_sidebar: default
---

# Gobernanza de datos y control de acceso

Usa *alias protegidos* para representar etapas clave de tu *pipeline* de desarrollo de modelos. Solo los *Administradores del Registro de Modelos* pueden añadir, modificar o eliminar alias protegidos. Los administradores del registro de modelos pueden definir y usar alias protegidos. W&B impide que los usuarios que no son administradores añadan o eliminen alias protegidos de las versiones de modelos.

:::info
Solo los administradores del equipo o los actuales administradores del registro pueden gestionar la lista de administradores del registro.
:::

Por ejemplo, supongamos que estableces `staging` y `producción` como alias protegidos. Cualquier miembro de tu equipo puede añadir nuevas versiones de modelos. Sin embargo, solo los administradores pueden añadir un alias `staging` o `producción`.

## Configurar el control de acceso
Los siguientes pasos describen cómo configurar los controles de acceso para el registro de modelos de tu equipo.

1. Navega a la aplicación W&B Model Registry en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Selecciona el botón de engranaje en la parte superior derecha de la página.
![](/images/models/rbac_gear_button.png)
3. Selecciona el botón **Gestionar administradores del registro**.
4. Dentro de la pestaña **Miembros**, selecciona los usuarios a los que deseas otorgar acceso para añadir y eliminar alias protegidos de las versiones de modelos.
![](/images/models/access_controls_admins.gif)

## Añadir alias protegidos
1. Navega a la aplicación W&B Model Registry en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Selecciona el botón de engranaje en la parte superior derecha de la página.
![](/images/models/rbac_gear_button.png)
3. Desplázate hacia abajo hasta la sección **Alias Protegidos**.
4. Haz clic en el icono de más (**+**) para añadir un nuevo alias.
![](/images/models/access_controls_add_protected_aliases.gif)