---
title: Hugging Face Diffusers
menu:
  default:
    identifier: ko-guides-integrations-diffusers
    parent: integrations
weight: 120
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" >}}

[Hugging Face Diffusers](https://huggingface.co/docs/diffusers)는 이미지, 오디오, 그리고 분자의 3D 구조까지 생성할 수 있는 최신 사전학습된 확산 모델을 위한 대표적인 라이브러리입니다. W&B 인테그레이션을 사용하면 풍부하고 유연한 실험 추적, 미디어 시각화, 파이프라인 아키텍처, 그리고 설정 관리를 간편하게 사용할 수 있으며, 대시보드를 통해 인터랙티브하게 중앙 집중식으로 모니터링할 수 있습니다.

## 단 두 줄로 완성되는 고급 로그 추적

실험에 사용된 모든 프롬프트, 네거티브 프롬프트, 생성된 미디어, 그리고 설정을 오직 두 줄의 코드만으로 자동으로 로그할 수 있습니다. 다음은 로그를 시작하기 위한 2줄의 코드입니다:

```python
# autolog 함수를 import 합니다
from wandb.integration.diffusers import autolog

# 파이프라인 호출 전에 autolog를 호출합니다
autolog(init=dict(project="diffusers_logging"))
```

| {{< img src="/images/integrations/diffusers-autolog-4.gif" alt="Experiment results logging" >}} | 
|:--:| 
| **실험 결과가 로그되는 예시입니다.** |

## 시작하기

1. `diffusers`, `transformers`, `accelerate`, 그리고 `wandb`를 설치하세요.

    - 커맨드라인:

        ```shell
        pip install --upgrade diffusers transformers accelerate wandb
        ```

    - 노트북:

        ```bash
        !pip install --upgrade diffusers transformers accelerate wandb
        ```

2. `autolog`으로 W&B Run을 초기화하고 [지원되는 모든 파이프라인 호출](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)의 입력값과 결과값을 자동으로 추적하세요.

    `autolog()` 함수는 `init` 파라미터를 받을 수 있으며, 이는 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에서 요구하는 파라미터를 가진 딕셔너리입니다.

    `autolog()`를 호출하면 W&B Run이 자동으로 시작되고, [지원되는 모든 파이프라인 호출](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)의 입력과 출력을 자동으로 추적합니다.

    - 각 파이프라인 호출은 워크스페이스 내에서 각각의 [테이블]({{< relref path="/guides/models/tables/" lang="ko" >}})로 기록되고, 해당 파이프라인 호출에 연결된 설정은 해당 run의 설정 워크플로우 리스트에 추가됩니다.
    - 프롬프트, 네거티브 프롬프트, 그리고 생성된 미디어는 [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ko" >}})에 로그됩니다.
    - 실험에 연결된 기타 모든 설정(예: seed 값, 파이프라인 아키텍처)은 run의 config 섹션에 저장됩니다.
    - 각 파이프라인 호출로 생성된 미디어도 run의 [미디어 패널]({{< relref path="/guides/models/track/log/media" lang="ko" >}})에 자동 기록됩니다.

    {{% alert %}}
    [지원되는 파이프라인 호출 목록](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)을 참고하실 수 있습니다. 만약 새로운 기능 요청이나 버그 리포팅이 필요하다면, [W&B GitHub 이슈 페이지](https://github.com/wandb/wandb/issues)에 이슈를 남겨주세요.
    {{% /alert %}}

## 예제

### Autologging

아래는 autolog 기능을 실제로 사용하는 간단한 엔드투엔드 예시입니다:

{{< tabpane text=true >}}
{{% tab header="Script" value="script" %}}
```python
import torch
from diffusers import DiffusionPipeline

# autolog 함수 import
from wandb.integration.diffusers import autolog

# 파이프라인 호출 전 autolog 호출
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인 초기화
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 네거티브 프롬프트, 시드 정의
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 이미지를 생성하는 파이프라인 호출
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)
```
{{% /tab %}}

{{% tab header="Notebook" value="notebook"%}}
```python
import torch
from diffusers import DiffusionPipeline

import wandb

# autolog 함수 import
from wandb.integration.diffusers import autolog

run = wandb.init()

# 파이프라인 호출 전 autolog 호출
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인 초기화
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 네거티브 프롬프트, 시드 정의
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 이미지를 생성하는 파이프라인 호출
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 실험 종료
run.finish()
```
{{% /tab %}}
{{< /tabpane >}}


- 단일 실험의 결과:

    {{< img src="/images/integrations/diffusers-autolog-2.gif" alt="Experiment results logging" >}}

- 여러 실험의 결과:

    {{< img src="/images/integrations/diffusers-autolog-1.gif" alt="Experiment results logging" >}}

- 실험의 설정 값:

    {{< img src="/images/integrations/diffusers-autolog-3.gif" alt="Experiment config logging" >}}

{{% alert %}}
IPython 노트북 환경에서 파이프라인을 사용한 뒤에는 반드시 [`wandb.Run.finish()`]({{< relref path="/ref/python/sdk/functions/finish.md" lang="ko" >}})를 명시적으로 호출해야 합니다. 파이썬 스크립트를 실행할 경우에는 필요하지 않습니다.
{{% /alert %}}

### 멀티-파이프라인 워크플로우 추적

이 섹션에서는 [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) 워크플로우에서 autolog를 사용하는 방법을 보여줍니다. 여기서는 [`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)이 생성한 latents를 refiner가 추가적으로 다듬는 방식입니다.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb" >}}

{{< tabpane text=true >}}

{{% tab header="Python Script" value="script" %}}

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

# SDXL base 파이프라인 초기화
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL refiner 파이프라인 초기화
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

# 무작위성을 제어해서 실험을 재현할 수 있게 만듭니다.
# 시드는 WandB에 자동으로 기록됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers용 WandB Autolog 호출. 프롬프트, 생성 이미지, 파이프라인 아키텍처,
# 각종 실험 설정 값이 W&B에 자동로그되어
# 이미지 생성 실험을 쉽게 재현, 공유, 분석할 수 있습니다.
autolog(init=dict(project="sdxl"))

# base 파이프라인으로 latents 생성
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# refiner 파이프라인으로 이미지를 다듬기
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner,
).images[0]
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

import wandb
from wandb.integration.diffusers import autolog

run = wandb.init()

# SDXL base 파이프라인 초기화
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL refiner 파이프라인 초기화
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

# 무작위성을 제어해서 실험을 재현할 수 있게 만듭니다.
# 시드는 WandB에 자동으로 기록됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers용 WandB Autolog 호출. 프롬프트, 생성 이미지, 파이프라인 아키텍처,
# 각종 실험 설정 값이 W&B에 자동로그되어
# 이미지 생성 실험을 쉽게 재현, 공유, 분석할 수 있습니다.
autolog(init=dict(project="sdxl"))

# base 파이프라인으로 latents 생성
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# refiner 파이프라인으로 이미지를 다듬기
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner,
).images[0]

# 실험 종료
run.finish()
```

{{% /tab %}}

{{< /tabpane >}}

- Stable Diffusion XL + Refiner 실험 예시:
    {{< img src="/images/integrations/diffusers-autolog-6.gif" alt="Stable Diffusion XL experiment tracking" >}}

## 추가 자료

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)