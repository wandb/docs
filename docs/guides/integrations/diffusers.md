---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Diffusers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/pixart-alpha-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers) is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. The W&B integration adds rich, flexible experiment tracking, media visualization and configuration managaement to interactive centralized dashboards without compromising that ease of use.

## Next-level logging in just 2 lines

Log all the prompts, negative prompts, generated media, and configs associated with your experiment by simply including 2 lines of code.

```python
import torch
from diffusers import DiffusionPipeline

# import the autolog function
from wandb.integration.diffusers import autolog

# call the autologger before calling the pipeline
autolog(init=dict(project="diffusers_logging"))

# Initialize the diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# Define the prompts, negative prompts, and seed.
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# call the pipeline to generate the images
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)
```

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **An example of how the results of your experiment are logged.** |

## Getting Started

First, we need to install `diffusers`, `transformers`, `accelerate`, and `wandb`.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade diffusers transformers accelerate wandb
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install --upgrade diffusers transformers accelerate wandb
```

  </TabItem>
</Tabs>

## Weights & Biases Autologger for Diffusers

This section demonstrates a typical text-conditional image generation workflow using the [`LatentConsistencyModelPipeline`](https://huggingface.co/docs/diffusers/v0.23.1/en/api/pipelines/latent_consistency_models). The autologger automatically logs the prompts, negative prompts, generated images, along with all associated configs for the experiment to Weights & Biases.

```python
import torch
from diffusers import DiffusionPipeline
from wandb.integration.diffusers import autolog


# call the autologger before calling the pipeline
autolog(init=dict(project="diffusers_logging"))

# Initialize the diffusion pipeline for latent consistency model
pipeline = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipeline = pipeline.to(torch_device="cuda", torch_dtype=torch.float32)

# Define the prompts, negative prompts, and seed.
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
generator = torch.Generator(device="cpu").manual_seed(10)

# call the pipeline to generate the images
images = pipeline(
    prompt,
    num_images_per_prompt=2,
    generator=generator,
    num_inference_steps=10,
)
```

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **An example of how the results of your experiment are logged.** |

:::info
The arguments passed to the `autolog` function is just `init` which accepts a dictionary of keywork arguments that would be passed to [`wandb.init()`](https://docs.wandb.ai/ref/python/init).
:::

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **An example of how the results of multiple experiments are logged in your workspace.** |

| ![An example of how the autologger logs the configs of your experiment](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **An example of how the autologger logs the configs of your experiment.** |

:::info
You can find a list of supported pipeline calls [here](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L67). In case, you want to request a new feature of this integration or report a bug associated with it, please open an issue on [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues).
:::

## More Resources

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)
