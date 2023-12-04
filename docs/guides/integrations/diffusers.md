---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hugging Face Diffusers

🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you’re looking for a simple inference solution or want to train your own diffusion model, 🤗 Diffusers is a modular toolbox that supports both. 🤗 Diffusers is designed with a focus on [usability over performance](https://huggingface.co/docs/diffusers/v0.23.1/en/conceptual/philosophy#usability-over-performance), [simple over easy](https://huggingface.co/docs/diffusers/v0.23.1/en/conceptual/philosophy#simple-over-easy), and [customizability over abstractions](https://huggingface.co/docs/diffusers/v0.23.1/en/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

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

This section demonstrates a typical text-conditional image generation workflow using the [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/text2img). The autologger automatically logs the prompts, negative prompts, generated images, along with all associated configs for the experiment to Weights & Biases.

```python
import torch
from diffusers import DiffusionPipeline
from wandb.integration.diffusers import autolog


# Initialize the diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Define the prompts, negative prompts, and seed.
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# call the autologger before calling the pipeline
autolog(init=dict(project="diffusers_logging"))

# call the pipeline to generate the images
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
    guidance_rescale=0.0,
)
```

| ![This gif shows how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **This gif shows how the results of your experiment are logged.** |

| ![This gif shows how your workspace would look with multiple experiments](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **This gif shows how your workspace would look with multiple experiments.** |

| ![This gif shows how the autologger logs the configs of your experiment](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **This gif shows how the autologger logs the configs of your experiment.** |

You can find a list of supported pipeline calls [here](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L67). In case, you want to request a new feature of this integration or report a bug associated with it, please open an issue on [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues).

## More Resources

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)
