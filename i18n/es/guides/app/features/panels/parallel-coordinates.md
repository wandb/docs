---
description: Compare results across machine learning experiments
displayed_sidebar: default
---

# Coordenadas Paralelas

Los gráficos de coordenadas paralelas resumen la relación entre un gran número de hiperparámetros y métricas de modelos de un vistazo.

![](/images/app_ui/parallel_coordinates.gif)

* **Ejes**: Diferentes hiperparámetros de [`wandb.config`](../../../../guides/track/config.md) y métricas de [`wandb.log`](../../../../guides/track/log/intro.md).
* **Líneas**: Cada línea representa un único run. Pasa el mouse sobre una línea para ver un tooltip con detalles sobre el run. Todas las líneas que coincidan con los filtros actuales se mostrarán, pero si apagas el ojo, las líneas se atenuarán.

## Configuración del Panel

Configura estas características en la configuración del panel— haz clic en el botón de editar en la esquina superior derecha del panel.

* **Tooltip**: Al pasar el mouse, aparece una leyenda con información sobre cada run
* **Títulos**: Edita los títulos de los ejes para que sean más legibles
* **Gradiente**: Personaliza el gradiente para que tenga cualquier rango de color que te guste
* **Escala logarítmica**: Cada eje se puede configurar para ver en una escala logarítmica de manera independiente
* **Invertir eje**: Cambia la dirección del eje— esto es útil cuando tienes tanto precisión como pérdida como columnas

[Véalo en vivo →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)