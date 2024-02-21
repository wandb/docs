---
description: Get Slack notifications when a new model version is linked to the model
  registry.
displayed_sidebar: default
---

# Crear alertas y notificaciones

<!-- # Notificaciones para nuevas versiones de modelo -->
Recibe notificaciones de Slack cuando una nueva versión de modelo se vincule al registro de modelos.


1. Navega a la aplicación W&B Registro de Modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Selecciona el modelo registrado del cual deseas recibir notificaciones.
3. Haz clic en el botón **Conectar Slack**.
    ![](/images/models/connect_to_slack.png)
4. Sigue las instrucciones para habilitar W&B en tu espacio de trabajo de Slack que aparecen en la página de OAuth.


Una vez que hayas configurado las notificaciones de Slack para tu equipo, puedes seleccionar los modelos registrados de los cuales deseas recibir notificaciones.

:::info
Un interruptor que dice **Nueva versión de modelo vinculada a...** aparece en lugar de un botón **Conectar Slack** si ya tienes configuradas las notificaciones de Slack para tu equipo.
:::

La captura de pantalla a continuación muestra un modelo registrado clasificador FMNIST que tiene notificaciones de Slack.

![](/images/models/conect_to_slack_fmnist.png)

Un mensaje se publica automáticamente en el canal de Slack conectado cada vez que una nueva versión de modelo se vincula al modelo registrado clasificador FMNIST.