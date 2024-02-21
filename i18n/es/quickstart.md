---
description: W&B Quickstart.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Inicio rápido

Instala W&B y comienza a hacer seguimiento de tus experimentos de aprendizaje automático en minutos.

## 1. Crea una cuenta e instala W&B
Antes de comenzar, asegúrate de crear una cuenta e instalar W&B:

1. [Regístrate](https://wandb.ai/site) para obtener una cuenta gratuita en [https://wandb.ai/site](https://wandb.ai/site) y luego inicia sesión en tu cuenta de wandb.  
2. Instala la biblioteca wandb en tu máquina en un entorno de Python 3 usando [`pip`](https://pypi.org/project/wandb/).  


Los siguientes fragmentos de código demuestran cómo instalar e iniciar sesión en W&B usando la CLI de W&B y la biblioteca de Python:

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Línea de Comandos', value: 'cli'},
  ]}>
  <TabItem value="cli">

Instala la CLI y la biblioteca de Python para interactuar con la API de Weights and Biases:

```bash
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Instala la CLI y la biblioteca de Python para interactuar con la API de Weights and Biases:


```notebook
!pip install wandb
```


  </TabItem>
</Tabs>

## 2. Inicia sesión en W&B


<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Línea de Comandos', value: 'cli'},
  ]}>
  <TabItem value="cli">

A continuación, inicia sesión en W&B:

```bash
wandb login
```

O si estás usando [W&B Server](./guides/hosting) (incluyendo **Nube Dedicada** o **Autogestionado**):

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

Si es necesario, solicita el nombre de host a tu administrador de despliegue.

Proporciona [tu clave API](https://wandb.ai/authorize) cuando se te solicite.

  </TabItem>
  <TabItem value="notebook">

A continuación, importa el SDK de Python de W&B e inicia sesión:

```python
wandb.login()
```

Proporciona [tu clave API](https://wandb.ai/authorize) cuando se te solicite.
  </TabItem>
</Tabs>

## 3. Inicia un run y haz seguimiento de hiperparámetros

Inicializa un objeto Run de W&B en tu script de Python o notebook con [`wandb.init()`](./ref/python/run.md) y pasa un diccionario al parámetro `config` con pares clave-valor de nombres de hiperparámetros y valores:

```python
run = wandb.init(
    # Establece el proyecto donde se registrará este run
    project="mi-proyecto-asombroso",
    # Haz seguimiento de hiperparámetros y metadatos del run
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```


Un [run](./guides/runs) es el bloque de construcción básico de W&B. Los usarás a menudo para [hacer seguimiento de métricas](./guides/track), [crear registros](./guides/artifacts), [crear trabajos](./guides/launch), y más.

## Poniéndolo todo junto

Poniéndolo todo junto, tu script de entrenamiento podría parecerse al siguiente ejemplo de código. El código resaltado muestra el código específico de W&B. 
Nota que hemos añadido código que imita el entrenamiento de aprendizaje automático.

```python
# train.py
import wandb
import random  # para el script de demo

# highlight-next-line
wandb.login()

epochs = 10
lr = 0.01

# highlight-start
run = wandb.init(
    # Establece el proyecto donde se registrará este run
    project="mi-proyecto-asombroso",
    # Haz seguimiento de hiperparámetros y metadatos del run
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# simulando un run de entrenamiento
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

¡Eso es todo! Navega a la App de W&B en [https://wandb.ai/home](https://wandb.ai/home) para ver cómo las métricas que registramos con W&B (precisión y pérdida) mejoraron durante cada paso de entrenamiento.

![Muestra la pérdida y precisión que se rastrearon cada vez que ejecutamos el script anterior. ](/images/quickstart/quickstart_image.png)

La imagen anterior (haz clic para ampliar) muestra la pérdida y precisión que se rastrearon cada vez que ejecutamos el script anterior. Cada objeto run que se creó se muestra dentro de la columna **Runs**. Cada nombre de run se genera aleatoriamente.

## ¿Qué sigue?

Explora el resto del ecosistema de W&B.

1. Revisa [Integraciones de W&B](guides/integrations) para aprender cómo integrar W&B con tu marco de ML como PyTorch, biblioteca de ML como Hugging Face, o servicio de ML como SageMaker. 
2. Organiza runs, incrusta y automatiza visualizaciones, describe tus hallazgos y comparte actualizaciones con colaboradores con [Reportes de W&B](./guides/reports).
2. Crea [Artefactos de W&B](./guides/artifacts) para hacer seguimiento de datasets, modelos, dependencias y resultados a través de cada paso de tu pipeline de aprendizaje automático.
3. Automatiza la búsqueda de hiperparámetros y explora el espacio de modelos posibles con [Barridos de W&B](./guides/sweeps).
4. Entiende tus datasets, visualiza predicciones de modelos y comparte ideas en un [panel de control central](./guides/tables).


![](/images/quickstart/wandb_demo_experiments.gif)

## Preguntas Comunes

**¿Dónde encuentro mi clave API?**
Una vez que hayas iniciado sesión en www.wandb.ai, la clave API estará en la página de [Autorización](https://wandb.ai/authorize).

**¿Cómo uso W&B en un entorno automatizado?**
Si estás entrenando modelos en un entorno automatizado donde es inconveniente ejecutar comandos de shell, como CloudML de Google, deberías ver nuestra guía de configuración con [Variables de Entorno](guides/track/environment-variables).

**¿Ofrecen instalaciones locales, on-prem?**
Sí, puedes [alojar W&B de forma privada](guides/hosting/) localmente en tus propias máquinas o en una nube privada, intenta [este notebook de tutorial rápido](http://wandb.me/intro) para ver cómo. Nota, para iniciar sesión en el servidor local de wandb puedes [establecer el host flag](guides/hosting/how-to-guides/basic-setup) a la dirección de la instancia local.  

**¿Cómo desactivo temporalmente el logging de wandb?**
Si estás probando código y quieres desactivar la sincronización de wandb, establece la variable de entorno [`WANDB_MODE=offline`](./guides/track/environment-variables).