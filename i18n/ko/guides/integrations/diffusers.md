---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Diffusers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers)는 이미지, 오디오, 심지어 분자의 3D 구조를 생성하기 위한 최신 사전학습된 확산 모델을 위한 주요 라이브러리입니다. Weights & Biases 통합은 사용의 용이성을 저해하지 않으면서 풍부하고 유연한 실험 추적, 미디어 시각화, 파이프라인 아키텍처, 그리고 대시보드에서의 대화형 중앙 집중식 설정 관리를 추가합니다.

## 단 두 줄로 하는 차세대 로깅

실험과 관련된 모든 프롬프트, 부정적 프롬프트, 생성된 미디어 및 설정을 단지 2줄의 코드를 포함하여 로그합니다. 로깅을 시작하는 데 필요한 2줄의 코드는 다음과 같습니다:

```python
# autolog 함수를 import합니다
from wandb.integration.diffusers import autolog

# 파이프라인을 호출하기 전에 autolog를 호출합니다
autolog(init=dict(project="diffusers_logging"))
```

| ![실험 결과가 어떻게 로그되는지에 대한 예시](@site/static/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **실험 결과가 어떻게 로그되는지에 대한 예시입니다.** |

## 시작하기

먼저, `diffusers`, `transformers`, `accelerate`, 그리고 `wandb`를 설치해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '커맨드라인', value: 'script'},
    {label: '노트북', value: 'notebook'},
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

### `autolog`는 어떻게 작동하나요?

`autolog()` 함수는 [`wandb.init()`](https://docs.wandb.ai/ref/python/init)에 필요한 파라미터를 받는 사전을 매개변수로 하는 `init`을 호출할 수 있습니다.

`autolog()`가 호출되면, [모든 지원되는 파이프라인 호출](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72)에서 입력과 출력을 자동으로 추적하는 Weights & Biases run이 초기화됩니다.

- 각 파이프라인 호출은 워크스페이스의 자체 [table](https://docs.wandb.ai/guides/tables)로 추적되며, 파이프라인 호출과 관련된 설정은 해당 run의 워크플로우 목록에 추가됩니다.
- 프롬프트, 부정적 프롬프트, 그리고 생성된 미디어는 [`wandb.Table`](https://docs.wandb.ai/guides/tables)에 로그됩니다.
- 실험과 관련된 모든 기타 설정을 포함하여 시드 및 파이프라인 아키텍처는 run의 설정 섹션에 저장됩니다.
- 각 파이프라인 호출에 대해 생성된 미디어도 run의 [미디어 패널](https://docs.wandb.ai/guides/track/log/media)에 로그됩니다.

:::info
지원되는 파이프라인 호출 목록은 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72)에서 확인할 수 있습니다. 이 통합의 새로운 기능을 요청하거나 관련 버그를 보고하려면, [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues)에서 문제를 열어주세요.
:::

다음은 autolog가 작동하는 간략한 엔드투엔드 예시입니다:

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python 스크립트', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import torch
from diffusers import DiffusionPipeline

# autolog 함수를 import합니다
from wandb.integration.diffusers import autolog

# 파이프라인을 호출하기 전에 autolog를 호출합니다
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인을 초기화합니다
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 부정적 프롬프트, 그리고 시드를 정의합니다.
prompt = [
    "말을 타고 있는 우주비행사의 사진",
    "용의 사진"
]
negative_prompt = ["못생긴, 기형적인", "못생긴, 기형적인"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 이미지를 생성하기 위해 파이프라인을 호출합니다
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
# autolog 함수를 import합니다
from wandb.integration.diffusers import autolog

# 파이프라인을 호출하기 전에 autolog를 호출합니다
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인을 초기화합니다
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 부정적 프롬프트, 그리고 시드를 정의합니다.
prompt = [
    "말을 타고 있는 우주비행사의 사진",
    "용의 사진"
]
negative_prompt = ["못생긴, 기형적인", "못생긴, 기형적인"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 이미지를 생성하기 위해 파이프라인을 호출합니다
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 실험을 마칩니다
wandb.finish()
```

  </TabItem>
</Tabs>

| ![실험 결과가 어떻게 로그되는지에 대한 예시](@site/static/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **실험 결과가 어떻게 로그되는지에 대한 예시입니다.** |

| ![실험 결과가 어떻게 로그되는지에 대한 예시](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **여러 실험의 결과가 워크스페이스에 어떻게 로그되는지에 대한 예시입니다.** |

| ![실험의 설정이 어떻게 autolog에 의해 로그되는지에 대한 예시](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **실험의 설정이 어떻게 autolog에 의해 로그되는지에 대한 예시입니다.** |

:::info
파이프라인을 호출한 후 IPython 노트북 환경에서 코드를 실행할 때는 [`wandb.finish()`](https://docs.wandb.ai/ref/python/finish)를 명시적으로 호출해야 합니다. 이는 파이썬 스크립트를 실행할 때는 필요하지 않습니다.
:::

## 다중 파이프라인 워크플로우 추적

이 섹션은 [`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)에 의해 생성된 레이턴트가 해당 리파이너에 의해 세련되는 전형적인 [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) 워크플로우에서의 autolog를 시연합니다.

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb"></CTAButtons>

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python 스크립트', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

# SDXL 기본 파이프라인을 초기화합니다
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL 리파이너 파이프라인을 초기화합니다
refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base_pipeline.text_encoder_2,
    vae=base_pipeline.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner_pipeline.enable_model_cpu_offload()

prompt = "화성에서 말을 타고 있는 우주비행사의 사진"
negative_prompt = "정적, 프레임, 그림, 일러스트레이션, sd 캐릭터, 저품질, 저해상도, 회색조, 단색, 코, 잘린, 저해상도, jpeg 아티팩트, 변형된 홍채, 변형된 동공, 나쁜 눈, 준리얼리스틱 최악의 품질, 나쁜 입술, 변형된 입, 변형된 얼굴, 변형된 손가락, 변형된 발가락, 가만히 서서, 포즈"

# 실험을 재현 가능하게 만들기 위해 무작위성을 제어합니다.
# 시드는 자동으로 WandB에 로그됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers를 위한 WandB Autolog를 호출합니다. 이것은 자동으로
# 프롬프트, 생성된 이미지, 파이프라인 아키텍처 및 모든
# 관련 실험 설정을 Weights & Biases에 로그하므로, 이미지 생성
# 실험을 쉽게 재현, 공유 및 분석할 수 있습니다.
autolog(init=dict(project="sdxl"))

# 레이턴트를 생성하기 위해 기본 파이프라인을 호출합니다
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# 세련된 이미지를 생성하기 위해 리파이너 파이프라인을 호출합니다
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

# SDXL 기본 파이프라인을 초기화합니다
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL 리파이너 파이프라인을 초기화합니다
refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base_pipeline.text_encoder_2,
    vae=base_pipeline.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner_pipeline.enable_model_cpu_offload()

prompt = "화성에서 말을 타고 있는 우주비행사의 사진"
negative_prompt = "정적, 프레임, 그림, 일러스트레이션, sd 캐릭터, 저품질, 저해상도, 회색조, 단색, 코, 잘린, 저해상도, jpeg 아티팩트, 변형된 홍채, 변형된 동공, 나쁜 눈, 준리얼리스틱 최악의 품질, 나쁜 입술, 변형된 입, 변형된 얼굴, 변형된 손가락, 변형된 발가락, 가만히 서서, 포즈"

# 실험을 재현 가능하게 만들기 위해 무작위성을 제어합니다.
# 시드는 자동으로 WandB에 로그됩니다.
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers를 위한 WandB Autolog를 호출합니다. 이것은 자동으로
# 프롬프트, 생성된 이미지, 파이프라인 아키텍처 및 모든
# 관련 실험 설정을 Weights & Biases에 로그하므로, 이미지 생성
# 실험을 쉽게 재현, 공유 및 분석할 수 있습니다.
autolog(init=dict(project="sdxl"))

# 레이턴트를 생성하기 위해 기본 파이프라인을 호출합니다
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# 세련된 이미지를 생성하기 위해 리파이너 파이프라인을 호출합니다
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner
).images[0]

# 실험을 마칩니다
wandb.finish()
```

  </TabItem>
</Tabs>

| ![autolog가 Stable Diffusion XL + Refiner 실험을 추적하는 예시](@site/static/images/integrations/diffusers-autolog-6.gif) | 
|:--:| 
| **autolog가 Stable Diffusion XL + Refiner 실험을 추적하는 예시입니다.** |

## 추가 자료

* [Stable Diffusion을 위한 프롬프트 엔지니어링 가이드](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: 텍스트 투 이미지 생성을 위한 확산 변환 모델](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)