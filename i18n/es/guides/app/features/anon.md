---
description: Log and visualize data without a W&B account
displayed_sidebar: default
---

# Modo Anónimo

¿Estás publicando código que quieres que cualquiera pueda ejecutar fácilmente? Utiliza el Modo Anónimo para permitir que alguien ejecute tu código, vea un panel de control de W&B y visualice resultados sin necesidad de crear una cuenta de W&B primero.

Permite que los resultados se registren en Modo Anónimo con `wandb.init(`**`anonymous="allow"`**`)`

:::info
**¿Publicando un artículo?** Por favor, [cita W&B](https://docs.wandb.ai/company/academics#bibtex-citation), y si tienes preguntas sobre cómo hacer tu código accesible mientras usas W&B, contáctanos en support@wandb.com.
:::

### ¿Cómo puede alguien sin cuenta ver los resultados?

Si alguien ejecuta tu script y has configurado `anonymous="allow"`:

1. **Crear cuenta temporal automáticamente:** W&B verifica si hay una cuenta ya iniciada. Si no hay cuenta, automáticamente creamos una nueva cuenta anónima y guardamos esa clave API para la sesión.
2. **Registrar resultados rápidamente:** El usuario puede ejecutar y re-ejecutar el script, y automáticamente ver los resultados aparecer en el panel de control de W&B. Estos runs anónimos no reclamados estarán disponibles durante 7 días.
3. **Reclamar datos cuando sean útiles**: Una vez que el usuario encuentre resultados valiosos en W&B, pueden fácilmente hacer clic en un botón en la pancarta en la parte superior de la página para guardar sus datos de run en una cuenta real. Si no reclaman un run, será eliminado después de 7 días.

:::caution
**Los enlaces de runs anónimos son sensibles**. Estos enlaces permiten que cualquiera vea y reclame los resultados de un experimento durante 7 días, así que asegúrate de solo compartir enlaces con personas de confianza. Si estás tratando de compartir resultados públicamente, pero ocultar la identidad del autor, por favor contáctanos en support@wandb.com para compartir más sobre tu caso de uso.
:::

### ¿Qué sucede con los usuarios que ya tienen cuentas existentes?

Si configuras `anonymous="allow"` en tu script, verificaremos primero para asegurarnos de que no haya una cuenta existente, antes de crear una cuenta anónima. Esto significa que si un usuario de W&B encuentra tu script y lo ejecuta, sus resultados se registrarán correctamente en su cuenta, justo como un run normal.

### ¿Qué características no están disponibles para los usuarios anónimos?

*   **No hay datos persistentes**: Los runs solo se guardan durante 7 días en una cuenta anónima. Los usuarios pueden reclamar los datos de un run anónimo guardándolos en una cuenta real.


![](@site/static/images/app_ui/anon_mode_no_data.png)

*   **No hay registro de artefactos**: Los runs imprimirán una advertencia en la línea de comandos de que no puedes registrar un artefacto en un run anónimo.

![](@site/static/images/app_ui/anon_example_warning.png)

* **No hay páginas de perfil o configuraciones**: Ciertas páginas no están disponibles en la UI, porque solo son útiles para cuentas reales.

## Ejemplo de uso

[Prueba el notebook de ejemplo](http://bit.ly/anon-mode) para ver cómo funciona el modo anónimo.

```python
import wandb

# Iniciar un run permitiendo cuentas anónimas
wandb.init(anonymous="allow")

# Registrar resultados de tu bucle de entrenamiento
wandb.log({"acc": 0.91})

# Marcar el run como finalizado
wandb.finish()
```