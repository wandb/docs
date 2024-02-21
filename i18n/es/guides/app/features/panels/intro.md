---
slug: /guides/app/features/panels
displayed_sidebar: default
---

# Paneles

Utiliza visualizaciones para explorar tus datos registrados, las relaciones entre hiperparámetros y métricas de salida, y ejemplos de datasets.

## Preguntas comunes

### Selecciono dimensiones en un gráfico de coordenadas paralelas y desaparece

Esto probablemente se deba a que tienes puntos en los nombres de tus parámetros de configuración. Aplanamos los parámetros anidados utilizando puntos, y solo manejamos 3 niveles de puntos en el backend. Recomiendo usar un carácter diferente como separador.

### Visualizar la máxima precisión en grupos

Activa el icono de "ojo" junto al mejor run de cada grupo para visualizar la máxima precisión en los gráficos.

![](/images/app_ui/visualize_max_accuracy.png)

### Descargar gráficos

Puedes descargar gráficos haciendo clic en la flecha hacia abajo y seleccionar un formato (.png, .svg, exportar API o exportar por CSV).

![](/images/app_ui/download_charts.png)