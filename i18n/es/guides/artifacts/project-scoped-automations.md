---
title: Artifact automations
description: Use an project scoped artifact automation in your project to trigger
  actions when aliases or versions in an artifact collection are created or changed.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Disparar eventos CI/CD con cambios en artefactos

Cree una automatización que se active cuando un artefacto cambia. Use automatizaciones de artefactos cuando desee automatizar acciones posteriores para el versionamiento de artefactos. Para crear una automatización, defina la [acción](#action-types) que desea que ocurra basada en un [tipo de evento](#event-types).

Algunos casos de uso comunes para automatizaciones que se disparan por cambios en un artefacto incluyen:

* Cuando se sube una nueva versión de un dataset de evaluación/retención, [disparar un trabajo de lanzamiento](#create-a-launch-automation) que realiza inferencia usando el mejor modelo de entrenamiento en el registro de modelos y crea un reporte con información de rendimiento.
* Cuando una nueva versión del dataset de entrenamiento es etiquetada como "producción", [disparar un trabajo de reentrenamiento](#create-a-launch-automation) con las configuraciones del modelo actual de mejor rendimiento.

:::info
Las automatizaciones de artefactos están limitadas a un proyecto. Esto significa que solo los eventos dentro de un proyecto activarán una automatización de artefacto.

Esto contrasta con las automatizaciones creadas en el Registro de Modelos de W&B. Las automatizaciones creadas en el registro de modelos están en el ámbito del Registro de Modelos; se activan cuando se realizan eventos en versiones de modelo vinculadas al [Registro de Modelos](../model_registry/intro.md). Para información sobre cómo crear automatizaciones para versiones de modelos, vea la página [Automatizaciones para CI/CD de Modelos](../model_registry/automation.md) en el capítulo del [Registro de Modelos](../model_registry/intro.md).
:::

## Tipos de evento
Un *evento* es un cambio que tiene lugar en el ecosistema de W&B. Puede definir dos tipos de eventos diferentes para colecciones de artefactos en su proyecto: **Se crea una nueva versión de un artefacto en una colección** y **Se agrega un alias de artefacto**.

:::tip
Use el tipo de evento **Se crea una nueva versión de un artefacto en una colección** para aplicar acciones recurrentes a cada versión de un artefacto. Por ejemplo, puede crear una automatización que automáticamente inicie un trabajo de entrenamiento cuando se crea una nueva versión de un artefacto de dataset.

Use el tipo de evento **Se agrega un alias de artefacto** para crear una automatización que se active cuando se aplica un alias específico a una versión de artefacto. Por ejemplo, podría crear una automatización que dispare una acción cuando alguien agrega el alias "control-de-calidad-del-conjunto-de-pruebas" a un artefacto, lo que luego activa el procesamiento posterior en ese dataset.
:::

## Tipos de acción
Una acción es una mutación receptiva (interna o externa) que ocurre como resultado de algún disparador. Hay dos tipos de acciones que puede crear en respuesta a eventos en colecciones de artefactos en su proyecto: webhooks y [Trabajos de Lanzamiento de W&B](../launch/intro.md).

* Webhooks: Comunicarse con un servidor web externo desde W&B con solicitudes HTTP.
* Trabajo de Lanzamiento de W&B: [Trabajos](../launch/create-launch-job.md) son plantillas de ejecución reutilizables y configurables que le permiten lanzar rápidamente nuevos [runs](../runs/intro.md) localmente en su escritorio o en recursos de cómputo externos como Kubernetes en EKS, Amazon SageMaker y más.


Las siguientes secciones describen cómo crear una automatización con webhooks y Lanzamiento de W&B.

## Crear una automatización de webhook 
Automatice un webhook basado en una acción con la interfaz de usuario de la App de W&B. Para hacer esto, primero establecerá un webhook, luego configurará la automatización de webhook.

### Agregar un secreto para autenticación o autorización
Los secretos son variables a nivel de equipo que le permiten ofuscar cadenas privadas como credenciales, claves API, contraseñas, tokens y más. W&B recomienda usar secretos para almacenar cualquier cadena que desee proteger el contenido de texto plano.

Para usar un secreto en su webhook, primero debe agregar ese secreto al administrador de secretos de su equipo.

:::info
* Solo los administradores de W&B pueden crear, editar o eliminar un secreto.
* Omita esta sección si el servidor externo al que envía solicitudes HTTP POST no usa secretos.
* Los secretos también están disponibles si usa [Servidor de W&B](../hosting/intro.md) en un despliegue de Azure, GCP o AWS. Conéctese con su equipo de cuenta de W&B para discutir cómo puede usar secretos en W&B si usa un tipo de despliegue diferente.
:::

Hay dos tipos de secretos que W&B sugiere que cree cuando usa una automatización de webhook:

* **Tokens de acceso**: Autorizar a los remitentes para ayudar a asegurar las solicitudes de webhook
* **Secreto**: Asegurar la autenticidad e integridad de los datos transmitidos desde las cargas útiles

Siga las instrucciones a continuación para crear un webhook:

1. Navegue a la interfaz de usuario de la App de W&B.
2. Haga clic en **Configuración del equipo**.
3. Desplácese hacia abajo en la página hasta encontrar la sección **Secretos del equipo**.
4. Haga clic en el botón **Nuevo secreto**.
5. Aparecerá un modal. Proporcione un nombre para su secreto en el campo **Nombre del secreto**.
6. Agregue su secreto en el campo **Secreto**.
7. (Opcional) Repita los pasos 5 y 6 para crear otro secreto (como un token de acceso) si su webhook requiere claves secretas adicionales o tokens para autenticar su webhook.

Especifique los secretos que desea usar para su automatización de webhook cuando configure el webhook. Vea la sección [Configurar un webhook](#configure-a-webhook) para más información.

:::tip
Una vez que crea un secreto, puede acceder a ese secreto en sus flujos de trabajo de W&B con `$`.
:::

### Configurar un webhook
Antes de poder usar un webhook, primero necesitará configurar ese webhook en la interfaz de usuario de la App de W&B.

:::info
* Solo los administradores de W&B pueden configurar un webhook para un equipo de W&B.
* Asegúrese de haber [creado uno o más secretos](#add-a-secret-for-authentication-or-authorization) si su webhook requiere claves secretas adicionales o tokens para autenticar su webhook.
:::

1. Navegue a la interfaz de usuario de la App de W&B.
2. Haga clic en **Configuración del equipo**.
4. Desplácese hacia abajo en la página hasta encontrar la sección **Webhooks**.
5. Haga clic en el botón **Nuevo webhook**.
6. Proporcione un nombre para su webhook en el campo **Nombre**.
7. Proporcione la URL del punto final para el webhook en el campo **URL**.
8. (Opcional) Desde el menú desplegable **Secreto**, seleccione el secreto que desea usar para autenticar la carga útil del webhook.
9. (Opcional) Desde el menú desplegable **Token de acceso**, seleccione el token de acceso que desea usar para autorizar al remitente.
9. (Opcional) Desde el menú desplegable **Token de acceso**, seleccione claves secretas adicionales o tokens requeridos para autenticar un webhook (como un token de acceso).

:::note
Vea la sección [Solucionar problemas de su webhook](#troubleshoot-your-webhook) para ver dónde se especifican el secreto y el token de acceso en la solicitud POST.
:::

### Agregar un webhook 
Una vez que tenga un webhook configurado y (opcionalmente) un secreto, navegue al espacio de trabajo de su proyecto. Haga clic en la pestaña **Automatizaciones** en la barra lateral izquierda.

1. Desde el menú desplegable **Tipo de evento**, seleccione un [tipo de evento](#event-types).
![](/images/artifacts/artifact_webhook_select_event.png)
2. Si seleccionó el evento **Se crea una nueva versión de un artefacto en una colección**, proporcione el nombre de la colección de artefactos a la que la automatización debe responder desde el menú desplegable **Colección de artefactos**.
![](/images/artifacts/webhook_new_version_artifact.png)
3. Seleccione **Webhooks** desde el menú desplegable **Tipo de acción**.
4. Haga clic en el botón **Siguiente paso**.
5. Seleccione un webhook desde el menú desplegable **Webhook**.
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. (Opcional) Proporcione una carga útil en el editor de expresión JSON. Vea la sección [Cargas útiles de ejemplo](#example-payloads) para ejemplos de casos de uso comunes.
7. Haga clic en **Siguiente paso**.
8. Proporcione un nombre para su automatización de webhook en el campo **Nombre de la automatización**.
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. (Opcional) Proporcione una descripción para su webhook.
10. Haga clic en el botón **Crear automatización**.

### Cargas útiles de ejemplo

Las siguientes pestañas demuestran cargas útiles de ejemplo basadas en casos de uso comunes. Dentro de los ejemplos, hacen referencia a las siguientes claves para referirse a objetos de condición en los parámetros de carga útil:
* `${event_type}` Se refiere al tipo de evento que disparó la acción.
* `${event_author}` Se refiere al usuario que disparó la acción.
* `${artifact_version}` Se refiere a la versión específica del artefacto que disparó la acción. Pasada como una instancia de artefacto.
* `${artifact_version_string}` Se refiere a la versión específica del artefacto que disparó la acción. Pasada como una cadena.
* `${artifact_collection_name}` Se refiere al nombre de la colección de artefactos a la que está vinculada la versión del artefacto.
* `${project_name}` Se refiere al nombre del proyecto que posee la mutación que disparó la acción.
* `${entity_name}` Se refiere al nombre de la entidad que posee la mutación que disparó la acción.


<Tabs
  defaultValue="github"
  values={[
    {label: 'Despacho de repositorio de GitHub', value: 'github'},
    {label: 'Notificación de Microsoft Teams', value: 'microsoft'},
    {label: 'Notificaciones de Slack', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
Verifique que sus tokens de acceso tengan el conjunto de permisos requeridos para disparar su flujo de trabajo GHA. Para más información, [vea estos Docs de GitHub](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event).
:::

  Envíe un despacho de repositorio desde W&B para disparar una acción de GitHub. Por ejemplo, suponga que tiene un flujo de trabajo que acepta un despacho de repositorio como disparador para la clave `on`:

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  La carga útil para el repositorio podría ser algo así como:

  ```json
  {
    "event_type": "BUILD_AND_DEPLOY",
    "client_payload": 
    {
      "event_author": "${event_author}",
      "artifact_version": "${artifact_version}",
      "artifact_version_string": "${artifact_version_string}",
      "artifact_collection_name": "${artifact_collection_name}",
      "project_name": "${project_name}",
      "entity_name": "${entity_name}"
      }
  }

  ```

:::note
La clave `event_type` en la carga útil del webhook debe coincidir con el campo `types` en el archivo YAML del flujo de trabajo de GitHub.
:::

  Los contenidos y la posición de las cadenas de plantilla renderizadas dependen del evento o versión del modelo para el que está configurada la automatización. `${event_type}` se renderizará como "LINK_ARTIFACT" o "ADD_ARTIFACT_ALIAS". Vea abajo para un ejemplo de mapeo:

  ```json
  ${event_type} --> "LINK_ARTIFACT" o "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<usuario-wandb>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entidad>/<project_name>/<nombre_artifacto>:<alias>"
  ${artifact_collection_name} --> "<nombre_colección_artifacto>"
  ${project_name} --> "<nombre_proyecto>"
  ${entity_name} --> "<entidad>"
  ```

  Use cadenas de plantilla para pasar dinámicamente contexto de W&B a GitHub Actions y otras herramientas. Si esas herramientas pueden llamar a scripts de Python, pueden consumir artefactos de W&B a través de la [API de W&B](../artifacts/download-and-use-an-artifact.md).

  Para más información sobre el despacho de repositorios, vea la [documentación oficial en el GitHub Marketplace](https://github.com/marketplace/actions/repository-dispatch).

  </TabItem>
  <TabItem value="microsoft">

  Configure un 'Webhook entrante' para obtener la URL del webhook para su canal de Teams configurando. El siguiente es un ejemplo de carga útil:
  
  ```json 
  {
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "summary": "Nueva notificación",
  "sections": [
    {
      "activityTitle": "Notificación de WANDB",
      "text": "Este es un mensaje de ejemplo enviado a través del webhook de Teams.",
      "facts": [
        {
          "name": "Autor",
          "value": "${event_author}"
        },
        {
          "name": "Tipo de Evento",
          "value": "${event_type}"
        }
      ],
      "markdown": true
    }
  ]
  }
  ```
  Puede usar cadenas de plantilla para inyectar datos de W&B en su carga útil en el momento de la ejecución (como se muestra en el ejemplo de Teams arriba).


  </TabItem>
  <TabItem value="slack">

  Configure su aplicación de Slack y agregue una integración de webhook entrante con las instrucciones destacadas en la [documentación de la API de Slack](https://api.slack.com/messaging/webhooks). Asegúrese de tener el secreto especificado bajo `Bot User OAuth Token` como el token de acceso de su webhook de W&B.
  
  El siguiente es un ejemplo de carga útil:

  ```json
    {
        "text": "¡Nueva alerta de WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Evento de artefacto: ${event_type}"
            }
        },
            {
                "type":"section",
                "text": {
                "type": "mrkdwn",
                "text": "Nueva versión: ${artifact_version_string}"
            }
            },
            {
            "type": "divider"
        },
            {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Autor: ${event_author}"
            }
            }
        ]
    }
  ```

  </TabItem>
</Tabs>

### Solucionar problemas de su webhook

El siguiente script bash genera una solicitud POST similar a la solicitud POST que W&B envía a su automatización de webhook cuando se activa.

Copie y pegue el código a continuación en un script de shell para solucionar problemas de su webhook. Especifique sus propios valores para lo siguiente:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`


```sh title="webhook_test.sh"
#!/bin/bash

# Su token de acceso y secreto
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# Los datos que desea enviar (por ejemplo, en formato JSON)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generar la firma HMAC
# Por seguridad, Wandb incluye el X-Wandb-Signature en el encabezado calculado 
# desde la carga útil y la clave secreta compartida asociada con el webhook 
# usando el algoritmo HMAC con SHA-256.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# Hacer la solicitud cURL
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

## Crear una automatización de lanzamiento
Inicie automáticamente un Trabajo de W&B.

:::info
Esta sección asume que ya ha creado un trabajo, una cola y tiene un agente activo sondeando. Para más información, vea la [documentación de Lanzamiento de W&B](../launch/intro.md).
:::


1. Desde el menú desplegable **Tipo de evento**, seleccione un tipo de evento. Vea la sección [Tipo de evento](#event-types) para información sobre eventos compatibles.
2. (Opcional) Si seleccionó el evento **Se crea una nueva versión de un