---
displayed_sidebar: default
---

# Gráfico de dispersión

Utiliza el gráfico de dispersión para comparar múltiples runs y visualizar cómo están desempeñándose tus experimentos. Hemos agregado algunas características personalizables:

1. Traza una línea a lo largo del mínimo, máximo y promedio
2. Herramientas de metadatos personalizados
3. Controlar los colores de los puntos
4. Establecer rangos de ejes
5. Cambiar los ejes a escala logarítmica

Aquí hay un ejemplo de la precisión de validación de diferentes modelos a lo largo de un par de semanas de experimentación. La herramienta está personalizada para incluir el tamaño del lote y el abandono, así como los valores en los ejes. También hay una línea que traza el promedio en curso de la precisión de validación.  
[Ver un ejemplo en vivo →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

![](https://paper-attachments.dropbox.com/s_9D642C56E99751C2C061E55EAAB63359266180D2F6A31D97691B25896D2271FC_1579031258748_image.png)

## Preguntas Comunes

### ¿Es posible trazar el máximo de una métrica en lugar de hacerlo paso a paso?

La mejor manera de hacer esto es crear un Gráfico de Dispersión de la métrica, entrar en el menú de Edición y seleccionar Anotaciones. Desde allí puedes trazar el máximo en curso de los valores