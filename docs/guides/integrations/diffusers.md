---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Hugging Face Diffusers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb"></CTAButtons>

[🤗 Diffusers](https://huggingface.co/docs/diffusers) は、最先端の学習済み拡散モデルを用いて画像、音声、さらには分子の3D構造を生成するためのライブラリです。W&Bのインテグレーションにより、リッチで柔軟な実験管理、メディア可視化、パイプラインアーキテクチャ、設定管理をインタラクティブで集中管理されたダッシュボードに追加し、使いやすさを損ないません。

## たった2行で次世代のログ

実験に関連する全てのプロンプト、ネガティブプロンプト、生成メディア、設定を、たった2行のコードを追加するだけでログに記録できます。以下の2行のコードを追加するだけでログを開始できます：

```python
# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))
```

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-4.gif) | 
|:--:| 
| **An example of how the results of your experiment are logged.** |

## はじめに

まず、`diffusers`、`transformers`、`accelerate` そして `wandb` をインストールする必要があります。

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

### `autolog` はどう機能しますか？

`autolog()` 関数は `init` パラメータとともに呼び出すことができ、このパラメータは [`wandb.init()`](https://docs.wandb.ai/ref/python/init) が必要とするパラメータの辞書を受け付けます。

`autolog()` が呼び出されると、Weights & Biases の run が初期化され、[サポートされているすべてのパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72) からの入力と出力が自動的にトラックされます。

- 各パイプライン呼び出しはワークスペース内の独自の[テーブル](https://docs.wandb.ai/guides/tables)に記録され、パイプライン呼び出しに関連する設定はその run の設定のワークフローリストに追加されます。
- プロンプト、ネガティブプロンプト、および生成されたメディアは [`wandb.Table`](https://docs.wandb.ai/guides/tables) に記録されます。
- シードやパイプラインアーキテクチャを含む実験に関連する他のすべての設定は、その run の設定セクションに保存されます。
- 各パイプライン呼び出しで生成されたメディアも run 内の[メディアパネル](https://docs.wandb.ai/guides/track/log/media)に記録されます。

:::info
サポートされているパイプライン呼び出しのリストは [こちら](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autolog.py#L12-L72) からご覧いただけます。新機能リクエストやバグ報告については、[https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues) にissueを提出してください。
:::

ここに、実際に autolog が動作している簡単なエンドツーエンドの例を示します：

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

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# パイプラインを呼び出して画像を生成
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
# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = [
    "a photograph of an astronaut riding a horse",
    "a photograph of a dragon"
]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# パイプラインを呼び出して画像を生成
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 実験を終了
wandb.finish()
```

  </TabItem>
</Tabs>

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-2.gif) | 
|:--:| 
| **An example of how the results of your experiment are logged.** |

| ![An example of how the results of your experiment are logged](@site/static/images/integrations/diffusers-autolog-1.gif) | 
|:--:| 
| **An example of how the results of multiple experiments are logged in your workspace.** |

| ![An example of how the autolog logs the configs of your experiment](@site/static/images/integrations/diffusers-autolog-3.gif) | 
|:--:| 
| **An example of how the autolog logs the configs of your experiment.** |

:::info
IPython ノートブック環境でコードを実行する場合、パイプライン呼び出し後に [`wandb.finish()`](https://docs.wandb.ai/ref/python/finish) を明示的に呼び出す必要があります。Python スクリプトを実行する場合は必要ありません。
:::

## マルチパイプラインワークフローの追跡

このセクションでは、[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) によって生成された潜在変数が対応するリファイナーによって改良される、典型的な [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) ワークフローでの autolog の使用例を示します。

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

# SDXL ベースパイプラインを初期化
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL リファイナーパイプラインを初期化
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

# 実験の再現性を高めるためにランダム性を制御します。
# シードは自動的に WandB に記録されます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers用の WandB Autolog を呼び出します。これにより、プロンプト、生成された画像、パイプラインアーキテクチャ、および関連するすべての実験設定が Weights & Biases に自動的に記録されるため、画像生成実験が再現可能で共有されやすく、分析しやすくなります。
autolog(init=dict(project="sdxl"))

# ベースパイプラインを呼び出して潜在変数を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを呼び出して改良された画像を生成
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

# SDXL ベースパイプラインを初期化
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL リファイナーパイプラインを初期化
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

# 実験の再現性を高めるためにランダム性を制御します。
# シードは自動的に WandB に記録されます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers用の WandB Autolog を呼び出します。これにより、プロンプト、生成された画像、パイプラインアーキテクチャ、および関連するすべての実験設定が Weights & Biases に自動的に記録されるため、画像生成実験が再現可能で共有されやすく、分析しやすくなります。
autolog(init=dict(project="sdxl"))

# ベースパイプラインを呼び出して潜在変数を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを呼び出して改良された画像を生成
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner
).images[0]

# 実験を終了
wandb.finish()
```

  </TabItem>
</Tabs>

| ![An example of how the autolog tracks an Stable Diffusion XL + Refiner experiment](@site/static/images/integrations/diffusers-autolog-6.gif) | 
|:--:| 
| **An example of how the autolog tracks an Stable Diffusion XL + Refiner experiment.** |

## 追加リソース

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)