---
title: Hugging Face Diffusers
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers)는 최첨단 사전학습된 확산 모델을 사용하여 이미지, 오디오, 심지어 3D 구조의 분자를 생성하는 데 최적의 라이브러리입니다. W&B 인테그레이션은 풍부하고 유연한 실험 추적, 미디어 시각화, 파이프라인 아키텍처 및 설정 관리를 인터랙티브한 중앙 집중 대시보드에 추가하여 사용의 용이성을 손상하지 않습니다.

## 단 두 줄로 다음 단계의 로깅

두 줄의 코드만 포함하여 실험과 관련된 모든 프롬프트, 네거티브 프롬프트, 생성된 미디어, 설정을 로그합니다. 로그를 시작하는 두 줄의 코드는 다음과 같습니다:

```python
# autolog 함수를 가져옵니다
from wandb.integration.diffusers import autolog

# autolog을 호출하여 파이프라인 호출 전에 실행합니다
autolog(init=dict(project="diffusers_logging"))
```

| ![실험 결과가 로그되는 예시](/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **실험 결과가 로그되는 예시입니다.** |

## 시작하기

먼저 `diffusers`, `transformers`, `accelerate`, `wandb`를 설치해야 합니다.

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

### `autolog`은 어떻게 작동하나요?

`autolog()` 함수는 `init` 파라미터와 함께 호출할 수 있으며, 이는 [`wandb.init()`](/ref/python/init)에서 요구하는 파라미터 사전을 받아들입니다.

`autolog()`가 호출되면 Weights & Biases run이 초기화되어, [지원되는 모든 파이프라인 호출](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72)로부터 입력과 출력을 자동으로 추적합니다.

- 각 파이프라인 호출은 워크스페이스 내의 자체 [테이블](/guides/tables)에서 추적되며, 파이프라인 호출과 관련된 설정은 그 run의 워크플로우 리스트에 추가됩니다.
- 프롬프트, 네거티브 프롬프트, 생성된 미디어는 [`wandb.Table`](/guides/tables)에 로그됩니다.
- 실험과 관련된 모든 다른 설정, 시드 및 파이프라인 아키텍처는 run의 설정 섹션에 저장됩니다.
- 각 파이프라인 호출에 대한 생성된 미디어는 run의 [미디어 패널](/guides/track/log/media)에도 로그됩니다.

:::안내
지원되는 파이프라인 호출 목록은 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72)에서 찾을 수 있습니다. 이 인테그레이션의 새로운 기능을 요청하거나 관련된 버그를 보고하려면 [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues)에 이슈를 열어주세요.
:::

여기 autolog의 엔드투엔드 예시가 있습니다:

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

# autolog 함수를 가져옵니다
from wandb.integration.diffusers import autolog

# autolog을 호출하여 파이프라인 호출 전에 실행합니다
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인 초기화
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 네거티브 프롬프트 및 시드 정의
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 파이프라인을 호출하여 이미지를 생성합니다
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)
```

  </TabItem>
  <TabItem value="notebook">

```python
import torch
from diffusers import DiffusionPipeline

import wandb
# autolog 함수를 가져옵니다
from wandb.integration.diffusers import autolog

# autolog을 호출하여 파이프라인 호출 전에 실행합니다
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인 초기화
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 네거티브 프롬프트 및 시드 정의
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 파이프라인을 호출하여 이미지를 생성합니다
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 실험을 마무리합니다
wandb.finish()
```

  </TabItem>
</Tabs>

| ![실험 결과가 로그되는 예시](/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **실험 결과가 로그되는 예시입니다.** |

| ![여러 실험의 결과가 워크스페이스에 로그되는 예시](/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **여러 실험의 결과가 워크스페이스에 로그되는 예시입니다.** |

| ![autolog가 실험의 설정을 로그하는 방법의 예시](/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **autolog가 실험의 설정을 로그하는 방법의 예시입니다.** |

:::안내
IPython 노트북 환경에서 파이프라인 호출 후 코드를 실행할 때는 [`wandb.finish()`](/ref/python/finish)을 명시적으로 호출해야 합니다. 이는 파이썬 스크립트를 실행할 때는 필요하지 않습니다.
:::

## 다중 파이프라인 워크플로우 추적

이 섹션에서는 [`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)이 생성한 잠금층을 해당 리파이너가 정제하는 일반적인 [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) 워크플로우와 함께 autolog을 시연합니다.

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb"></CTAButtons>

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

# SDXL 베이스 파이프라인 초기화
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL 리파이너 파이프라인 초기화
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

# 실험을 재현 가능하게 하기 위해 무작위성을 제어합니다.
# 시드는 WandB에 자동으로 로그됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# WandB Autolog을 Diffusers로 호출합니다. 이는 자동으로 프롬프트,
# 생성된 이미지, 파이프라인 아키텍처 및 관련 실험 설정을
# Weights & Biases에 로그하여, 이미지 생성 실험을 쉽게 재현, 공유 그리고
# 분석할 수 있도록 합니다.
autolog(init=dict(project="sdxl"))

# 베이스 파이프라인을 호출하여 잠금층을 생성합니다
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# 리파이너 파이프라인을 호출하여 정제된 이미지를 생성합니다
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

# SDXL 베이스 파이프라인 초기화
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL 리파이너 파이프라인 초기화
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

# 실험을 재현 가능하게 하기 위해 무작위성을 제어합니다.
# 시드는 WandB에 자동으로 로그됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# WandB Autolog을 Diffusers로 호출합니다. 이는 자동으로 프롬프트,
# 생성된 이미지, 파이프라인 아키텍처 및 관련 실험 설정을
# Weights & Biases에 로그하여, 이미지 생성 실험을 쉽게 재현, 공유 그리고
# 분석할 수 있도록 합니다.
autolog(init=dict(project="sdxl"))

# 베이스 파이프라인을 호출하여 잠금층을 생성합니다
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# 리파이너 파이프라인을 호출하여 정제된 이미지를 생성합니다
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner
).images[0]

# 실험을 마무리합니다
wandb.finish()
```

  </TabItem>
</Tabs>

| ![Stable Diffusion XL + Refiner 실험을 autolog가 추적하는 방법의 예시](/images/integrations/diffusers-autolog-6.gif) | 
|:--:| 
| **Stable Diffusion XL + Refiner 실험을 autolog가 추적하는 방법의 예시입니다.** |

## 추가 자료

* [Stable Diffusion을 위한 프롬프트 엔지니어링 가이드](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: 텍스트 투 이미지 생성을 위한 확산 트랜스포머 모델](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)