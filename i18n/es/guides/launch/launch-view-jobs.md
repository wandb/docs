---
displayed_sidebar: default
---

# Ver trabajos de lanzamiento

La siguiente página describe cómo ver información sobre trabajos de lanzamiento agregados a colas.

## Ver trabajos

Vea trabajos agregados a una cola con la aplicación W&B.

1. Navegue a la aplicación W&B en https://wandb.ai/home.
2. Seleccione **Lanzamiento** dentro de la sección **Aplicaciones** de la barra lateral izquierda.
3. Seleccione el desplegable **Todas las entidades** y seleccione la entidad a la que pertenece el trabajo de lanzamiento.
4. Expanda la UI colapsable desde la página de la aplicación de lanzamiento para ver una lista de trabajos agregados a esa cola específica.

:::info
Un run se crea cuando el agente de lanzamiento ejecuta un trabajo de lanzamiento. En otras palabras, cada run listado corresponde a un trabajo específico que fue agregado a esa cola.
:::

Por ejemplo, la siguiente imagen muestra dos runs que fueron creados a partir de un trabajo llamado `job-source-launch_demo-canonical`. El trabajo fue agregado a una cola llamada `Start queue`. El primer run listado en la cola se llama `resilient-snowball` y el segundo run listado se llama `earthy-energy-165`.


![](/images/launch/launch_jobs_status.png)

Dentro de la UI de la aplicación W&B, puede encontrar información adicional sobre runs creados a partir de trabajos de lanzamiento, como:
   - **Run**: El nombre del run de W&B asignado a ese trabajo.
   - **Job ID**: El nombre del trabajo.
   - **Proyecto**: El nombre del proyecto al que pertenece el run.
   - **Estado**: El estado del run en cola.
   - **Autor**: La entidad de W&B que creó el run.
   - **Fecha de creación**: La marca de tiempo cuando se creó la cola.
   - **Hora de inicio**: La marca de tiempo cuando comenzó el trabajo.
   - **Duración**: Tiempo, en segundos, que tomó completar el run del trabajo.

## Listar trabajos
Vea una lista de trabajos que existen dentro de un proyecto con el CLI de W&B. Use el comando de lista de trabajos de W&B y proporcione el nombre del proyecto y la entidad a la que pertenece el trabajo de lanzamiento con las banderas `--project` y `--entity`, respectivamente.

```bash
 wandb job list --entity tu-entidad --project nombre-del-proyecto
```

## Verificar el estado de un trabajo

La siguiente tabla define el estado que un run en cola puede tener:


| Estado | Descripción |
| --- | --- |
| **Idle** | El run está en una cola sin agentes activos. |
| **En cola** | El run está en cola esperando a que un agente lo procese. |
| **Pendiente** | El run ha sido recogido por un agente pero aún no ha comenzado. Esto podría deberse a que los recursos no están disponibles en el cluster. |
| **Ejecutando** | El run se está ejecutando actualmente. |
| **Terminado** | El trabajo fue terminado por el usuario. |
| **Estrellado** | El run dejó de enviar datos o no comenzó con éxito. |
| **Fallido** | El run terminó con un código de salida distinto de cero o el run falló al iniciar. |
| **Finalizado** | El trabajo se completó con éxito. |