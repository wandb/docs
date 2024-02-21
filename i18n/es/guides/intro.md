---
description: An overview of what is W&B along with links on how to get started if
  you are a first time user.
slug: /guides
displayed_sidebar: default
---

# ¿Qué es W&B?

Weights & Biases (W&B) es la plataforma para desarrolladores de IA, con herramientas para entrenar modelos, afinar modelos y aprovechar modelos base.

Configura W&B en 5 minutos, luego itera rápidamente en tu pipeline de aprendizaje automático con la confianza de que tus modelos y datos están rastreados y versionados en un sistema de registro confiable.

![](@site/static/images/general/architecture.png)

Este diagrama describe la relación entre los productos de W&B.

**[Modelos de W&B](/guides/models.md)** es un conjunto de herramientas ligeras e interoperables para profesionales de aprendizaje automático que entrenan y afinan modelos.
- [Experimentos](/guides/track/intro.md): Seguimiento de experimentos de aprendizaje automático
- [Registro de Modelos](/guides/model_registry/intro.md): Gestionar modelos en producción de manera centralizada
- [Launch](/guides/launch/intro.md): Escalar y automatizar cargas de trabajo
- [Barridos](/guides/sweeps/intro.md): Ajuste de hiperparámetros y optimización de modelos

**[Prompts de W&B](/guides/prompts/intro.md)** es para depurar y evaluar LLMs.

**[Plataforma W&B](/guides/platform.md)** es un conjunto de bloques de construcción poderosos para el seguimiento y visualización de datos y modelos, y la comunicación de resultados.
- [Artefactos](/guides/artifacts/intro.md): Versionar activos y rastrear linaje
- [Tablas](/guides/tables/intro.md): Visualizar y consultar datos tabulares
- [Reportes](/guides/reports/intro.md): Documentar y colaborar en tus descubrimientos
- [Weave](/guides/app/features/panels/weave) Consultar y crear visualizaciones de tus datos

## ¿Eres un usuario nuevo de W&B?

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Demostración End-to-End de Weights &amp; Biases" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Comienza a explorar W&B con estos recursos:

1. [Notebook Introductorio](http://wandb.me/intro): Ejecuta código de muestra rápida para rastrear experimentos en 5 minutos
2. [Inicio Rápido](../quickstart.md): Lee un resumen rápido de cómo y dónde agregar W&B a tu código
1. Explora nuestra [Guía de Integraciones](./integrations/intro.md) y nuestra lista de reproducción de YouTube [Integración Fácil con W&B](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) para información sobre cómo integrar W&B con tu marco de aprendizaje automático preferido.
1. Consulta la [Guía de Referencia de la API](../ref/README.md) para especificaciones técnicas sobre la Biblioteca Python de W&B, CLI y operaciones de Weave.

## ¿Cómo funciona W&B?

Te recomendamos leer las siguientes secciones en este orden si eres un usuario nuevo de W&B:

1. Aprende sobre [Runs](./runs/intro.md), la unidad básica de cómputo de W&B.
2. Crea y rastrea experimentos de aprendizaje automático con [Experimentos](./track/intro.md).
3. Descubre el bloque de construcción flexible y ligero de W&B para la versionamiento de datasets y modelos con [Artefactos](./artifacts/intro.md).
4. Automatiza la búsqueda de hiperparámetros y explora el espacio de modelos posibles con [Barridos](./sweeps/intro.md).
5. Gestiona el ciclo de vida del modelo desde el entrenamiento hasta la producción con [Gestión del Modelo](./model_registry/intro.md).
6. Visualiza predicciones a través de versiones de modelo con nuestra guía de [Visualización de Datos](./tables/intro.md).
7. Organiza Runs de W&B, incrusta y automatiza visualizaciones, describe tus hallazgos y comparte actualizaciones con colaboradores con [Reportes](./reports/intro.md).