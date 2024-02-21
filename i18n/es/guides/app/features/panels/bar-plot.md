---
description: Visualize metrics, customize axes, and compare categorical data as bars.
displayed_sidebar: default
---

# Gráfico de Barras

Un gráfico de barras presenta datos categóricos con barras rectangulares que pueden trazarse vertical u horizontalmente. Los gráficos de barras aparecen por defecto con **wandb.log()** cuando todos los valores registrados tienen una longitud de uno.

![Trazando gráficos de Caja y Barras horizontales en W&B](/images/app_ui/bar_plot.png)

Personaliza con ajustes del gráfico para limitar el máximo de runs a mostrar, agrupar runs por cualquier configuración y renombrar etiquetas.

![](/images/app_ui/bar_plot_custom.png)

### Personalizar Gráficos de Barras

También puedes crear Gráficos de **Caja** o **Violín** para combinar muchas estadísticas resumidas en un solo tipo de gráfico**.**

1. Agrupa runs mediante la tabla de runs.
2. Haz clic en 'Agregar panel' en el espacio de trabajo.
3. Añade un 'Gráfico de Barras' estándar y selecciona la métrica a trazar.
4. Bajo la pestaña 'Agrupación', elige 'gráfico de caja' o 'Violín', etc. para trazar cualquiera de estos estilos.

![Personalizar Gráficos de Barras](@site/static/images/app_ui/bar_plots.gif)