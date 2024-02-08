---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Diffusers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers) is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. The W&B integration adds rich, flexible experiment tracking, media visualization, pipeline architecture, and configuration managaement to interactive centralized dashboards without compromising that ease of use.

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

This section demonstrates a typical text-conditional image generation workflow using the [`LatentConsistencyModelPipeline`](https://huggingface.co/docs/diffusers/v0.23.1/en/api/pipelines/latent_consistency_models). The autologger automatically logs the prompts, negative prompts, generated images, pipeline architecture, along with all associated configs for the experiment to Weights & Biases.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

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

  </TabItem>
  <TabItem value="notebook">

```python
import torch
from diffusers import DiffusionPipeline

import wandb
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

# Finish the experiment
wandb.finish()
```

  </TabItem>
</Tabs>

Calling the `autolog` creates a [W&B run](https://docs.wandb.ai/guides/runs). All subsequent pipeline calls are therefore registered and their inputs and outputs are logged to the run. The prompts, negative prompts, and the generated media are logged in a [`wandb.Table`](https://docs.wandb.ai/guides/tables) while all other configs associated with the experiment including seed and the pipeline architecture are stored in the config section for the run. The generated media are also logged in [media panels](https://docs.wandb.ai/guides/track/log/media) in the run.

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **An example of how the results of your experiment are logged.** |

:::info
The arguments passed to the `autolog` function is just `init` which accepts a dictionary of keyword arguments that would be passed to [`wandb.init()`](https://docs.wandb.ai/ref/python/init).
:::

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **An example of how the results of multiple experiments are logged in your workspace.** |

| ![An example of how the autologger logs the configs of your experiment](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **An example of how the autologger logs the configs of your experiment.** |

## Tracking Multi-pipeline Workflows

This section demonstrates how the Weights & Biases autologger can be used for tracking workflows that involve multiple diffusion pipeline calls. We initially generate an image using the [`AmusedPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/amused) and then use [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img) to modify the generated image.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import torch
from diffusers import AmusedPipeline, StableDiffusionImg2ImgPipeline
from wandb.integration.diffusers import autolog

# call the autologger before calling the pipelines
autolog()

# Generate initial image using AmusedPipeline
pipe = AmusedPipeline.from_pretrained(
    "amused/amused-512", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(prompt, generator=generator).images[0]

# Use aMused generated image as input to Stable Diffusion img2img pipeline
image2image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
image2image_pipe = image2image_pipe.to("cuda")
prompt = "A fantasy landscape, trending on artstation"
init_image = image.resize((768, 512))
generator = torch.Generator(device="cuda").manual_seed(42)
images = image2image_pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,
    guidance_scale=7.5,
    generator=generator,
).images
```

  </TabItem>
  <TabItem value="notebook">

```python
import torch
from diffusers import AmusedPipeline, StableDiffusionImg2ImgPipeline

import wandb
from wandb.integration.diffusers import autolog

# call the autologger before calling the pipelines
autolog()

# Generate initial image using AmusedPipeline
pipe = AmusedPipeline.from_pretrained(
    "amused/amused-512", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(prompt, generator=generator).images[0]

# Use aMused generated image as input to Stable Diffusion img2img pipeline
image2image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
image2image_pipe = image2image_pipe.to("cuda")
prompt = "A fantasy landscape, trending on artstation"
init_image = image.resize((768, 512))
generator = torch.Generator(device="cuda").manual_seed(42)
images = image2image_pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,
    guidance_scale=7.5,
    generator=generator,
).images

# Finish the experiment
wandb.finish()
```

  </TabItem>
</Tabs>

| ![An example of how the autologger tracks multiple diffusion pipeline calls in a single experiment](@site/static/images/integrations/diffusers-autolog-5.png) | 
|:--:| 
| **An example of how the autologger tracks multiple diffusion pipeline calls in a single experiment.** |

:::info
When `autolog()` is called, it initializes a Weights & Biases run, which automatically tracks the inputs and the outputs from [all supported pipeline calls](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72). Each pipeline call is tracked into its own table in the workspace, and the configs associated with the pipeline call is appended to the list of workflows in the configs for that run.
:::

Let's demonstrate the autologger with a typical [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) workflow, in which the latents generated by the [`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) is refined by the corresponding refiner.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

# initialize the SDXL base pipeline
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# initialize the SDXL refiner pipeline
refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base_pipeline.text_encoder_2,
    vae=base_pipeline.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner_pipeline.enable_model_cpu_offload()

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing"

# Make the experiment reproducible by controlling randomness.
# The seed would be automatically logged to WandB.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Call WandB Autolog for Diffusers. This would automatically log
# the prompts, generated images, pipeline architecture and all
# associated experiment configs to Weights & Biases, thus making your
# image generation experiments easy to reproduce, share and analyze.
autolog(init=dict(project="sdxl"))

# Call the base pipeline to generate the latents
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# Call the refiner pipeline to generate the refined image
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner
).images[0]
```

  </TabItem>
  <TabItem value="notebook">

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

import wandb
from wandb.integration.diffusers import autolog

# initialize the SDXL base pipeline
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# initialize the SDXL refiner pipeline
refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base_pipeline.text_encoder_2,
    vae=base_pipeline.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner_pipeline.enable_model_cpu_offload()

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing"

# Make the experiment reproducible by controlling randomness.
# The seed would be automatically logged to WandB.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Call WandB Autolog for Diffusers. This would automatically log
# the prompts, generated images, pipeline architecture and all
# associated experiment configs to Weights & Biases, thus making your
# image generation experiments easy to reproduce, share and analyze.
autolog(init=dict(project="sdxl"))

# Call the base pipeline to generate the latents
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# Call the refiner pipeline to generate the refined image
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner
).images[0]

# Finish the experiment
wandb.finish()
```

  </TabItem>
</Tabs>

| ![An example of how the autologger tracks an SDXL + Refiner experiment](@site/static/images/integrations/diffusers-autolog-6.gif) | 
|:--:| 
| **An example of how the autologger tracks an SDXL + Refiner experiment.** |

:::info
You can find a list of supported pipeline calls [here](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72). In case, you want to request a new feature of this integration or report a bug associated with it, please open an issue on [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues).
:::

## More Resources

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)
