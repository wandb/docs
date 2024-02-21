---
description: Model registry to manage the model lifecycle from training to production
slug: /guides/model_registry
displayed_sidebar: default
---

# Registro de modelos
El Registro de Modelos de W&B alberga los modelos entrenados de un equipo donde los profesionales de ML pueden publicar candidatos para producción para ser consumidos por equipos y partes interesadas aguas abajo. Se utiliza para albergar modelos en etapa de candidatura y gestionar flujos de trabajo asociados con la puesta en escena.

![](/images/models/model_reg_landing_page.png)

Con el Registro de Modelos de W&B, puedes:

* [Marcar tus mejores versiones de modelo para cada tarea de aprendizaje automático.](./link-model-version.md)
* [Automatizar](./automation.md) procesos aguas abajo y CI/CD del modelo.
* Mover versiones del modelo a través de su ciclo de vida de ML; desde la puesta en escena hasta la producción.
* Seguir el linaje de un modelo y auditar el historial de cambios en los modelos de producción.

![](/images/models/models_landing_page.png)

## Cómo funciona
Rastrea y gestiona tus modelos en etapa con unos simples pasos.

1. **Registrar una versión del modelo**: En tu script de entrenamiento, añade unas pocas líneas de código para guardar los archivos del modelo como un artefacto en W&B.
2. **Comparar rendimiento**: Revisa gráficos en vivo para comparar las métricas y predicciones de muestra del entrenamiento y validación del modelo. Identifica cuál versión del modelo tuvo el mejor desempeño.
3. **Enlazar al registro**: Marca la mejor versión del modelo enlazándola a un modelo registrado, ya sea programáticamente en Python o de manera interactiva en la UI de W&B.

El siguiente fragmento de código demuestra cómo registrar y enlazar un modelo al Registro de Modelos:

```python showLineNumbers
import wandb
import random

# Iniciar un nuevo run de W&B
run = wandb.init(project="models_quickstart")

# Simular el registro de métricas del modelo
run.log({"acc": random.random()})

# Crear un archivo de modelo simulado
with open("my_model.h5", "w") as f:
    f.write("Modelo: " + str(random.random()))

# Registrar y enlazar el modelo al Registro de Modelos
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **Conectar las transiciones del modelo a flujos de trabajo de CI/CD**: transitar modelos candidatos a través de etapas de flujo de trabajo y [automatizar acciones aguas abajo](./automation.md) con webhooks o trabajos.

## Cómo empezar
Dependiendo de tu caso de uso, explora los siguientes recursos para comenzar con los Modelos de W&B:

* Revisa la serie de videos de dos partes:
  1. [Registro y registro de modelos](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [Consumir modelos y automatizar procesos aguas abajo](https://www.youtube.com/watch?v=8PFCrDSeHzw) en el Registro de Modelos.
* Lee el [recorrido por los modelos](./walkthrough.md) para un esquema paso a paso de los comandos del SDK de Python de W&B que podrías usar para crear, rastrear y usar un artefacto de dataset.
* Revisa [este](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) reporte sobre cómo el Registro de Modelos se ajusta a tu flujo de trabajo de ML y los beneficios de usar uno para la gestión del modelo.
* Aprende sobre:
   * [Modelos protegidos y control de acceso](./access_controls.md).
   * [Cómo conectar el Registro de Modelos a procesos de CI/CD](./automation.md).
   * Configurar [notificaciones de Slack](./notifications.md) cuando una nueva versión del modelo esté enlazada a un modelo registrado.