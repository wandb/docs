---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Diffusers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers)는 이미지, 오디오, 심지어 분자의 3D 구조를 생성하기 위한 최신 사전 훈련된 확산 모델을 위한 주요 라이브러리입니다. W&B 통합은 사용의 용이성을 해치지 않으면서 풍부하고 유연한 실험 추적, 미디어 시각화, 파이프라인 아키텍처 및 구성 관리를 대화형 중앙 대시보드에 추가합니다.

## 단 2줄로 차원이 다른 로깅

실험과 관련된 모든 프롬프트, 부정적 프롬프트, 생성된 미디어 및 구성을 단순히 2줄의 코드를 포함하여 로그합니다.

```python
import torch
from diffusers import DiffusionPipeline

# autolog 함수를 import합니다
from wandb.integration.diffusers import autolog

# pipeline을 호출하기 전에 autologger를 호출합니다
autolog(init=dict(project="diffusers_logging"))

# 확산 파이프라인을 초기화합니다
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# 프롬프트, 부정적 프롬프트 및 시드를 정의합니다.
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

| ![실험 결과가 어떻게 로그되는지의 예시입니다](@site/static/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **실험 결과가 어떻게 로그되는지의 예시입니다.** |

## 시작하기

먼저, `diffusers`, `transformers`, `accelerate`, 그리고 `wandb`를 설치해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '커맨드 라인', value: 'script'},
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

## Weights & Biases Autologger for Diffusers

이 섹션은 [`LatentConsistencyModelPipeline`](https://huggingface.co/docs/diffusers/v0.23.1/en/api/pipelines/latent_consistency_models)을 사용한 전형적인 텍스트 조건부 이미지 생성 워크플로를 보여줍니다. Autologger는 프롬프트, 부정적 프롬프트, 생성된 이미지, 파이프라인 아키텍처와 실험과 관련된 모든 구성을 Weights & Biases에 자동으로 로그합니다.

```python
import torch
from diffusers import DiffusionPipeline
from wandb.integration.diffusers import autolog


# pipeline을 호출하기 전에 autologger를 호출합니다
autolog(init=dict(project="diffusers_logging"))

# 잠재 일관성 모델의 확산 파이프라인을 초기화합니다
pipeline = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipeline = pipeline.to(torch_device="cuda", torch_dtype=torch.float32)

# 프롬프트, 부정적 프롬프트 및 시드를 정의합니다.
prompt = [
    "말을 타고 있는 우주비행사의 사진",
    "용의 사진"
]
generator = torch.Generator(device="cpu").manual_seed(10)

# 이미지를 생성하기 위해 파이프라인을 호출합니다
images = pipeline(
    prompt,
    num_images_per_prompt=2,
    generator=generator,
    num_inference_steps=10,
)
```

`autolog`를 호출하면 [W&B 실행](https://docs.wandb.ai/guides/runs)이 생성됩니다. 따라서 이후 모든 파이프라인 호출은 등록되며 그 입력과 출력은 실행에 로그됩니다. 프롬프트, 부정적 프롬프트, 생성된 미디어는 [`wandb.Table`](https://docs.wandb.ai/guides/tables)에 로그되며, 실험과 관련된 모든 다른 구성들은 실행의 구성 섹션에 저장됩니다. 생성된 미디어는 또한 실행의 [미디어 패널](https://docs.wandb.ai/guides/track/log/media)에 로그됩니다.

| ![실험 결과가 어떻게 로그되는지의 예시입니다](@site/static/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **실험 결과가 어떻게 로그되는지의 예시입니다.** |

:::info
`autolog` 함수에 전달된 인수는 [`wandb.init()`](https://docs.wandb.ai/ref/python/init)에 전달될 키워드 인수의 사전을 받는 `init`만 있습니다.
:::

| ![실험 결과가 어떻게 로그되는지의 예시입니다](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **워크스페이스에서 여러 실험의 결과가 어떻게 로그되는지의 예시입니다.** |

| ![autologger가 실험의 구성을 어떻게 로그하는지의 예시입니다](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **autologger가 실험의 구성을 어떻게 로그하는지의 예시입니다.** |

:::info
지원되는 파이프라인 호출 목록은 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L67)에서 확인할 수 있습니다. 이 통합의 새로운 기능을 요청하거나 관련 버그를 보고하고 싶다면, [https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues)에서 이슈를 열어주세요.
:::

## 추가 자료

* [안정적 확산을 위한 프롬프트 엔지니어링 가이드](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [텍스트 투 이미지 생성을 위한 확산 트랜스포머 모델 PIXART-α](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)