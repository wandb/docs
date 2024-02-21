---
description: Time to live policies (TTL)
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Gestiona la retención de datos con la política de tiempo de vida (TTL) de artefactos

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb"/>

Programa cuándo se eliminarán los artefactos de W&B con la política de tiempo de vida (TTL) de artefactos de W&B. Cuando eliminas un artefacto, W&B marca ese artefacto como un *borrado suave*. En otras palabras, el artefacto se marca para su eliminación, pero los archivos no se eliminan inmediatamente del almacenamiento. Para más información sobre cómo W&B elimina artefactos, consulta la página [Eliminar artefactos](./delete-artifacts.md).

Mira [este](https://www.youtube.com/watch?v=hQ9J6BoVmnc) video tutorial para aprender cómo gestionar la retención de datos con Artifacts TTL en la aplicación de W&B.

:::note
W&B desactiva la opción de establecer una política de TTL para artefactos de modelo vinculados al Registro de Modelos. Esto es para ayudar a asegurar que los modelos vinculados no expiren accidentalmente si se utilizan en flujos de trabajo de producción.
:::
:::info
* Solo los administradores de equipo pueden ver la [configuración del equipo](../app/settings-page/team-settings.md) y acceder a configuraciones de TTL a nivel de equipo como (1) permitir quién puede establecer o editar una política de TTL o (2) establecer un TTL predeterminado para el equipo.
* Si no ves la opción de establecer o editar una política de TTL en los detalles de un artefacto en la UI de la aplicación de W&B o si establecer un TTL programáticamente no cambia con éxito la propiedad TTL de un artefacto, tu administrador de equipo no te ha dado permiso para hacerlo.
:::

## Definir quién puede editar y establecer políticas de TTL
Define quién puede establecer y editar políticas de TTL dentro de un equipo. Puedes otorgar permisos de TTL solo a los administradores del equipo, o puedes otorgar permisos de TTL tanto a los administradores del equipo como a los miembros del equipo.

:::info
Solo los administradores de equipo pueden definir quién puede establecer o editar una política de TTL.
:::

1. Navega a la página de perfil de tu equipo.
2. Selecciona la pestaña **Configuración**.
3. Navega a la **sección de tiempo de vida (TTL) de artefactos**.
4. Desde el **menú desplegable de permisos de TTL**, selecciona quién puede establecer y editar políticas de TTL.
5. Haz clic en **Revisar y guardar configuración**.
6. Confirma los cambios y selecciona **Guardar configuración**.

![](/images/artifacts/define_who_sets_ttl.gif)

## Crear una política de TTL
Establece una política de TTL para un artefacto ya sea cuando creas el artefacto o de manera retroactiva después de que el artefacto haya sido creado.

Para todos los fragmentos de código a continuación, reemplaza el contenido envuelto en `<>` con tu información para usar el fragmento de código.

### Establecer una política de TTL al crear un artefacto
Usa el SDK de Python de W&B para definir una política de TTL cuando creas un artefacto. Las políticas de TTL se definen típicamente en días.

:::tip
Definir una política de TTL al crear un artefacto es similar a cómo normalmente [creas un artefacto](./construct-an-artifact.md). Con la excepción de que pasas un delta de tiempo al atributo `ttl` del artefacto.
:::

Los pasos son los siguientes:

1. [Crea un artefacto](./construct-an-artifact.md).
2. [Añade contenido al artefacto](./construct-an-artifact.md#add-files-to-an-artifact) como archivos, un directorio o una referencia.
3. Define un límite de tiempo de TTL con el tipo de dato [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) que es parte de la biblioteca estándar de Python.
4. [Registra el artefacto](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server).

El siguiente fragmento de código demuestra cómo crear un artefacto y establecer una política de TTL.

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<mi-nombre-de-proyecto>", entity="<mi-entidad>")
artifact = wandb.Artifact(name="<nombre-del-artefacto>", type="<tipo>")
artifact.add_file("<mi_archivo>")

artifact.ttl = timedelta(days=30)  # Establecer política de TTL
run.log_artifact(artifact)
```

El fragmento de código anterior establece la política de TTL para el artefacto a 30 días. En otras palabras, W&B elimina el artefacto después de 30 días.

### Establecer o editar una política de TTL después de crear un artefacto
Usa la UI de la aplicación de W&B o el SDK de Python de W&B para definir una política de TTL para un artefacto que ya existe.

:::note
Cuando modificas el TTL de un artefacto, el tiempo que el artefacto tarda en expirar se calcula aún usando la marca de tiempo `createdAt` del artefacto.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'SDK de Python', value: 'python'},
    {label: 'Aplicación W&B', value: 'app'},
  ]}>
  <TabItem value="python">

1. [Obtén tu artefacto](./download-and-use-an-artifact.md).
2. Pasa un delta de tiempo al atributo `ttl` del artefacto.
3. Actualiza el artefacto con el método [`save`](../../ref/python/run.md#save).


El siguiente fragmento de código muestra cómo establecer una política de TTL para un artefacto:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<mi-entidad/mi-proyecto/mi-artefacto:alias>")
artifact.ttl = timedelta(days=365 * 2)  # Eliminar en dos años
artifact.save()
```

El ejemplo de código anterior establece la política de TTL a dos años.

  </TabItem>
  <TabItem value="app">

1. Navega a tu proyecto de W&B en la UI de la aplicación de W&B.
2. Selecciona el icono de artefacto en el panel izquierdo.
3. De la lista de artefactos, expande el tipo de artefacto que 
4. Selecciona en la versión del artefacto para la cual quieres editar la política de TTL.
5. Haz clic en la pestaña **Versión**.
6. Desde el menú desplegable, selecciona **Editar política de TTL**.
7. Dentro del modal que aparece, selecciona **Personalizado** desde el menú desplegable de política de TTL.
8. Dentro del campo **Duración de TTL**, establece la política de TTL en unidades de días.
9. Selecciona el botón **Actualizar TTL** para guardar tus cambios.

![](/images/artifacts/edit_ttl_ui.gif)

  </TabItem>
</Tabs>

### Establecer políticas de TTL predeterminadas para un equipo

:::info
Solo los administradores de equipo pueden establecer una política de TTL predeterminada para un equipo.
:::

Establece una política de TTL predeterminada para tu equipo. Las políticas de TTL predeterminadas se aplican a todos los artefactos existentes y futuros basados en sus respectivas fechas de creación. Los artefactos con políticas de TTL a nivel de versión existentes no se ven afectados por la TTL predeterminada del equipo.

1. Navega a la página de perfil de tu equipo.
2. Selecciona la pestaña **Configuración**.
3. Navega a la **sección de tiempo de vida (TTL) de artefactos**.
4. Haz clic en **Establecer la política de TTL predeterminada del equipo**.
5. Dentro del campo **Duración**, establece la política de TTL en unidades de días.
6. Haz clic en **Revisar y guardar configuración**.
7. Confirma los cambios y luego selecciona **Guardar configuración**.

![](/images/artifacts/set_default_ttl.gif)

## Desactivar una política de TTL
Usa el SDK de Python de W&B o la UI de la aplicación de W&B para desactivar una política de TTL para una versión específica de artefacto.
<!-- 
:::note
Los artefactos con un TTL desactivado no heredarán el TTL de una colección de artefactos. Consulta (## Heredar Política de TTL) sobre cómo eliminar el TTL de un artefacto y heredar del nivel de colección TTL.
::: -->

<Tabs
  defaultValue="python"
  values={[
    {label: 'SDK de Python', value: 'python'},
    {label: 'Aplicación W&B', value: 'app'},
  ]}>
  <TabItem value="python">

1. [Obtén tu artefacto](./download-and-use-an-artifact.md).
2. Establece el atributo `ttl` del artefacto a `None`.
3. Actualiza el artefacto con el método [`save`](../../ref/python/run.md#save).


El siguiente fragmento de código muestra cómo desactivar una política de TTL para un artefacto:
```python
artifact = run.use_artifact("<mi-entidad/mi-proyecto/mi-artefacto:alias>")
artifact.ttl = None
artifact.save()
```


  </TabItem>
  <TabItem value="app">

1. Navega a tu proyecto de W&B en la UI de la aplicación de W&B.
2. Selecciona el icono de artefacto en el panel izquierdo.
3. De la lista de artefactos, expande el tipo de artefacto que 
4. Selecciona en la versión del artefacto para la cual quieres editar la política de TTL.
5. Haz clic en la pestaña de Versión.
6. Haz clic en el icono de menú de tres puntos al lado del botón **Vincular al registro**.
7. Desde el menú desplegable, selecciona **Editar política de TTL**.
8. Dentro del modal que aparece, selecciona **Desactivar** desde el menú desplegable de política de TTL.
9. Selecciona el botón **Actualizar TTL** para guardar tus cambios.

![](/images/artifacts/remove_ttl_polilcy.gif)

  </TabItem>
</Tabs>

## Ver políticas de TTL
Visualiza las políticas de TTL para artefactos con el SDK de Python o con la UI de la aplicación de W&B.

<Tabs
  defaultValue="python"
  values={[
    {label: 'SDK de Python', value: 'python'},
    {label: 'Aplicación W&B', value: 'app'},
  ]}>
  <TabItem value="python">

Usa una declaración de impresión para ver la política de TTL de un artefacto. El siguiente ejemplo muestra cómo recuperar un artefacto y ver su política de TTL:

```python
artifact = run.use_artifact("<mi-entidad/mi-proyecto/mi-artefacto:alias>")
print(artifact.ttl)
```

  </TabItem>
  <TabItem value="app">


Visualiza una política de TTL para un artefacto con la UI de la aplicación de W&B.

1. Navega a la aplicación de W&B en [https://wandb.ai](https://wandb.ai).
2. Ve a tu proyecto de W&B.
3. Dentro de tu proyecto, selecciona la pestaña de Artefactos en la barra lateral izquierda.
4. Haz clic en una colección.

Dentro de la vista de la colección puedes ver todos los artefactos en la colección seleccionada. Dentro de la columna `Tiempo de Vida` verás la política de TTL asignada a ese artefacto.

![](/images/artifacts/ttl_collection_panel_ui.png)