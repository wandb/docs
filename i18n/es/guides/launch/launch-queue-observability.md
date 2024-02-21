---
displayed_sidebar: default
---

# Panel de control de monitoreo de cola (beta)

Utiliza el **Panel de control de monitoreo de cola** interactivo para ver cuándo una cola de lanzamiento está en uso intensivo o inactiva, visualizar las cargas de trabajo que se están ejecutando y detectar trabajos ineficientes. El panel de control de la cola de lanzamiento es especialmente útil para decidir si estás utilizando de manera efectiva tu hardware de cómputo o recursos en la nube.

Para un análisis más profundo, la página enlaza al espacio de trabajo de seguimiento de experimentos de W&B y a proveedores externos de monitoreo de infraestructura como Datadog, NVIDIA Base Command o consolas de nube.

:::info
Los paneles de control de monitoreo de cola requieren W&B Weave. W&B Weave aún no está disponible en despliegues gestionados por el cliente o AWS/GCP Dedicated Cloud. Contacta a tu representante de W&B para aprender más.
:::

## Panel de control y gráficas
Usa la pestaña **Monitor** para ver la actividad de una cola que ocurrió durante los últimos siete días. Utiliza el panel izquierdo para controlar rangos de tiempo, agrupaciones y filtros.

El panel de control contiene varias gráficas que responden a preguntas comunes sobre rendimiento y eficiencia. Las secciones siguientes describen los elementos de la UI de los paneles de control de cola.

### Estado del trabajo
La gráfica de **Estado del trabajo** muestra cuántos trabajos están ejecutándose, pendientes, en cola o completados en cada intervalo de tiempo. Usa la gráfica de **Estado del trabajo** para identificar períodos de inactividad en la cola.

![](/images/launch/launch_obs_jobstatus.png)

Por ejemplo, supón que tienes un recurso fijo (como DGX BasePod). Si observas una cola inactiva con el recurso fijo, esto podría sugerir una oportunidad para ejecutar trabajos de lanzamiento pre-emptibles de menor prioridad como barridos.

Por otro lado, supón que utilizas un recurso en la nube y ves ráfagas periódicas de actividad. Las ráfagas periódicas de actividad podrían sugerir una oportunidad para ahorrar dinero reservando recursos para tiempos particulares.

A la derecha de la gráfica hay una clave que muestra qué colores representan el [estado de un trabajo de lanzamiento](./launch-view-jobs.md#check-the-status-of-a-job).

:::tip
Los elementos `En cola` podrían indicar oportunidades para trasladar cargas de trabajo a otras colas. Un pico en fallos puede identificar usuarios que podrían necesitar ayuda con su configuración de trabajo de lanzamiento.
:::

### Tiempo en cola

La gráfica de **Tiempo en cola** muestra la cantidad de tiempo (en segundos) que un trabajo de lanzamiento estuvo en una cola para una fecha o rango de tiempo dado.

![](/images/launch/launch_obs_queuedtime.png)

El eje x muestra un marco de tiempo que especificas y el eje y muestra el tiempo (en segundos) que un trabajo de lanzamiento estuvo en una cola de lanzamiento. Por ejemplo, supón que en un día dado hay 10 trabajos de lanzamiento en cola. La gráfica de **Tiempo en cola** muestra 600 segundos si esos 10 trabajos de lanzamiento esperan un promedio de 60 segundos cada uno.

:::tip
Usa la gráfica de **Tiempo en cola** para identificar a los usuarios afectados por largos tiempos de espera en la cola.
:::

Personaliza el color de cada trabajo con el control de `Agrupación` en la barra izquierda.

### Ejecuciones de trabajo

![](/images/launch/launch_obs_jobruns2.png)

Esta gráfica muestra el inicio y fin de cada trabajo ejecutado en un período de tiempo, con colores distintos para cada ejecución. Esto facilita ver de un vistazo qué cargas de trabajo estaba procesando la cola en un tiempo dado.

Usa la herramienta de Selección en la parte inferior derecha del panel para resaltar trabajos y poblar los detalles en la tabla de abajo.

### Uso de CPU y GPU
Usa las gráficas de **Uso de GPU por trabajo**, **Uso de CPU por trabajo**, **Memoria de GPU por trabajo** y **Memoria del sistema por trabajo** para ver la eficiencia de tus trabajos de lanzamiento.

![](/images/launch/launch_obs_gpu.png)

Por ejemplo, puedes usar la gráfica de **Memoria de GPU por trabajo** para ver si una ejecución de W&B tomó mucho tiempo en completarse y si usó o no un bajo porcentaje de sus núcleos de CPU.

El eje x de cada gráfica muestra la duración de una ejecución de W&B (creada por un trabajo de lanzamiento) en segundos. Pasa el mouse sobre un punto de datos para ver información sobre una ejecución de W&B como el ID de la ejecución, el proyecto al que pertenece la ejecución, el trabajo de lanzamiento que creó la ejecución de W&B y más.

### Errores

El panel de **Errores** muestra errores que ocurrieron en una cola de lanzamiento dada. Más específicamente, el panel de Errores muestra una marca de tiempo de cuándo ocurrió el error, el nombre del trabajo de lanzamiento de donde proviene el error, y el mensaje de error que se creó. Por defecto, los errores se ordenan del más reciente al más antiguo.

![](/images/launch/launch_obs_errors.png)

Usa el panel de **Errores** para identificar y desbloquear usuarios.

## Enlaces externos

La vista del panel de observabilidad de cola es consistente a través de todos los tipos de cola, pero en muchos casos, puede ser útil saltar directamente a monitores específicos del entorno. Para lograr esto, añade un enlace desde la consola directamente desde el panel de observabilidad de cola.

En la parte inferior de la página, haz clic en `Administrar enlaces` para abrir un panel. Añade la URL completa de la página que deseas. A continuación, añade una etiqueta. Los enlaces que añades aparecen en la sección de **Enlaces externos**.