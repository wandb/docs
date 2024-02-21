---
displayed_sidebar: default
---

# Documentar modelo de aprendizaje automático

Añade una descripción a la tarjeta del modelo de tu modelo registrado para documentar aspectos de tu modelo de aprendizaje automático. Algunos temas que vale la pena documentar incluyen:

* **Resumen**: Un resumen de qué es el modelo. El propósito del modelo. El framework de aprendizaje automático que utiliza el modelo, y así sucesivamente.
* **Datos de entrenamiento**: Describe los datos de entrenamiento utilizados, el procesamiento realizado en el conjunto de datos de entrenamiento, dónde se almacenan esos datos y demás.
* **Arquitectura**: Información sobre la arquitectura del modelo, capas y cualquier elección de diseño específica.
* **Deserializar el modelo**: Proporciona información sobre cómo alguien de tu equipo puede cargar el modelo en memoria.
* **Tarea**: El tipo específico de tarea o problema que el modelo de aprendizaje automático está diseñado para realizar. Es una categorización de la capacidad pretendida del modelo.
* **Licencia**: Los términos legales y permisos asociados con el uso del modelo de aprendizaje automático. Ayuda a los usuarios del modelo a entender el marco legal bajo el cual pueden utilizar el modelo.
* **Referencias**: Citas o referencias a investigaciones relevantes, datasets, o recursos externos.
* **Despliegue**: Detalles sobre cómo y dónde se despliega el modelo y orientación sobre cómo el modelo se integra en otros sistemas empresariales, como plataformas de orquestación de flujos de trabajo.

## Añadir una descripción a la tarjeta del modelo

1. Navega a la aplicación W&B Registro de Modelos en [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Selecciona **Ver detalles** al lado del nombre del modelo registrado para el cual quieres crear una tarjeta de modelo.
2. Ve a la sección **Tarjeta del modelo**.
![](/images/models/model_card_example.png)
3. Dentro del campo **Descripción**, proporciona información sobre tu modelo de aprendizaje automático. Formatea el texto dentro de una tarjeta de modelo con [lenguaje de marcado Markdown](https://www.markdownguide.org/).

Por ejemplo, las siguientes imágenes muestran la tarjeta del modelo de un modelo registrado de **Predicción de Default de Tarjeta de Crédito**.
![](/images/models/model_card_credit_example.png)