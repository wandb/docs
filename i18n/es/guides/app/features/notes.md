---
description: Add notes to your runs and projects, and use notes to describe your findings
  in reports
displayed_sidebar: default
---

# Notas

Hay algunas formas de tomar notas sobre tu trabajo en W&B.

1. Añadir notas a un run. Estas notas aparecen en la página del run en la pestaña de resumen y en la tabla de runs en la página del proyecto.
2. Añadir notas a un proyecto. Estas notas aparecen en la página del proyecto en la pestaña de resumen.
3. Añadir un panel de markdown en la página del run, la página del proyecto o la página del reporte.

## Añadir notas a un run específico

Puedes editar las notas de un run en dos lugares.

1. **Página del Proyecto**: la tabla tiene una columna de notas editable
2. **Página del Run**: la pestaña de resumen muestra información sobre un run, y puedes

En la página del proyecto, expande la tabla. Haz clic en "Añadir notas..." para escribir notas en línea.

![Editando notas en la tabla en la página del proyecto](https://downloads.intercomcdn.com/i/o/148296355/34114b47362b0378e233a440/2019-09-13+08.05.17.gif)

Este campo también aparece en la página individual del run. Haz clic en el nombre del run en la tabla para ir a la página del run. Haz clic en la pestaña superior del lado izquierdo para ir a la pestaña de Resumen. El campo tiene mucho más espacio para extenderse aquí. Puedes escribir tantas notas como quieras en este espacio, y se mostrará una vista previa en la tabla de runs cuando pases el mouse sobre el campo de notas.

![Editando notas en la pestaña de resumen en la página del run](https://downloads.intercomcdn.com/i/o/148297196/afdb48d2fb59aaa0c90c3aed/2019-09-13+08.06.45.gif)

También puedes crear un reporte para añadir gráficas y markdown lado a lado. Usa diferentes secciones para mostrar diferentes runs y contar una historia sobre lo que has trabajado. Estas notas se pueden guardar y compartir con colegas.

## Escribir notas descriptivas comparando runs

Usa reportes para escribir sobre tus hallazgos al comparar múltiples runs. Haz clic en "Añadir visualización" para añadir un panel de markdown. Puedes organizar estos paneles junto a paneles de gráficas.

![](https://downloads.intercomcdn.com/i/o/148297552/64e5baa86a48927158d17456/2019-09-13+08.08.31.gif)

## Escribir Markdown en un nuevo panel

Usa markdown y ecuaciones latex como:

```
$TPR = Sensibilidad = \dfrac{TP}{TP+FN}$
```

Haz esto añadiendo un panel, seleccionando markdown, y luego introduciendo tu texto en markdown, tablas ecuaciones y bloque de código se renderizarán automáticamente al hacer clic fuera del panel de markdown.

![](@site/static/images/app_ui/tables_panel.gif)