---
title: Hugging Face Diffusers
menu:
  default:
    identifier: ja-guides-integrations-diffusers
    parent: integrations
weight: 120
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" >}}

[Hugging Face Diffusers](https://huggingface.co/docs/diffusers) は、画像、音声、さらには分子の3D構造を生成するための最先端の学習済み拡散モデル用のライブラリです。W&B のインテグレーションにより、実験管理、メディア可視化、パイプラインアーキテクチャ、設定管理をインタラクティブな集中ダッシュボードに追加し、使いやすさを損なうことなく提供します。

## 次のレベルのログ記録をたった2行で

実験に関連するすべてのプロンプト、ネガティブプロンプト、生成されたメディア、設定を、たった2行のコードを含めるだけでログに記録します。ログを開始するための2行のコードはこちらです：

```python
# autolog 関数をインポートします
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出します
autolog(init=dict(project="diffusers_logging"))
```

| {{< img src="/images/integrations/diffusers-autolog-4.gif" alt="実験の結果がどのようにログに記録されるかの例" >}} | 
|:--:| 
| **実験の結果がどのようにログに記録されるかの例。** |

## 設定を開始する

1. `diffusers`、`transformers`、`accelerate`、`wandb` をインストールします。

    - コマンドライン：

        ```shell
        pip install --upgrade diffusers transformers accelerate wandb
        ```

    - ノートブック：

        ```bash
        !pip install --upgrade diffusers transformers accelerate wandb
        ```

2. `autolog` を使用して Weights & Biases の run を初期化し、[すべてのサポートされたパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) の入力と出力を自動的に追跡します。

    `init` パラメータを使用して `autolog()` 関数を呼び出すことができ、これは [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) に必要なパラメータの辞書を受け取ります。

    `autolog()` を呼び出すと、Weights & Biases の run を初期化し、[すべてのサポートされたパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) の入力と出力を自動的に追跡します。

    - 各パイプライン呼び出しは、ワークスペース内の個々の [テーブル]({{< relref path="/guides/core/tables/" lang="ja" >}}) に追跡され、パイプライン呼び出しに関連する設定は、その run のワークフローリストに追加されます。
    - プロンプト、ネガティブプロンプト、生成されたメディアは [`wandb.Table`]({{< relref path="/guides/core/tables/" lang="ja" >}}) にログされます。
    - シードやパイプラインアーキテクチャを含む実験に関連する他のすべての設定は、run の設定セクションに保存されます。
    - 各パイプライン呼び出しで生成されたメディアも、run の[メディアパネル]({{< relref path="/guides/models/track/log/media" lang="ja" >}}) にログされます。

    {{% alert %}}
    サポートされているパイプライン呼び出しのリストは [こちら](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) で見つけることができます。このインテグレーションの新しい機能をリクエストしたり、関連するバグを報告したい場合は、[https://github.com/wandb/wandb/issues](https://github.com/wandb/wandb/issues) で問題をオープンしてください。
    {{% /alert %}}

## 例

### オートログの使用

以下は、オートログが動作するエンドツーエンドの例です：

{{< tabpane text=true >}}
{{% tab header="スクリプト" value="script" %}}
```python
import torch
from diffusers import DiffusionPipeline

# autolog 関数をインポートします
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出します
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化します
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義します。
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# パイプラインを呼び出して画像を生成します
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)
```
{{% /tab %}}

{{% tab header="ノートブック" value="notebook"%}}
```python
import torch
from diffusers import DiffusionPipeline

import wandb

# autolog 関数をインポートします
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼び出します
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化します
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義します。
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# パイプラインを呼び出して画像を生成します
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 実験を終了します
wandb.finish()
```
{{% /tab %}}
{{< /tabpane >}}

- 単一の実験の結果：

    {{< img src="/images/integrations/diffusers-autolog-2.gif" alt="単一の実験結果がどのようにログに記録されるかの例" >}}

- 複数の実験の結果：

    {{< img src="/images/integrations/diffusers-autolog-1.gif" alt="複数の実験結果がどのようにログに記録されるかの例" >}}

- 実験の設定：

    {{< img src="/images/integrations/diffusers-autolog-3.gif" alt="オートログが実験の設定をどのように記録するかの例" >}}

{{% alert %}}
IPython ノートブック環境でコードを実行する際は、パイプラインを呼び出した後に明示的に [`wandb.finish()`]({{< relref path="/ref/python/finish" lang="ja" >}}) を呼び出す必要があります。Python スクリプトを実行している場合は必要ありません。
{{% /alert %}}

### マルチパイプラインワークフローの追跡

このセクションでは、[Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) ワークフローを使用したオートログの例を示します。このワークフローでは、[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) によって生成された潜在情報が、対応するリファイナによって精錬されます。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb" >}}

{{< tabpane text=true >}}

{{% tab header="Python スクリプト" value="script" %}}

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

# SDXL ベースパイプラインを初期化します
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL リファイナーパイプラインを初期化します
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

# 実験を再現可能にするため、ランダム性をコントロールします。
# シードは自動的に WandB にログされます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers に向けて WandB Autolog を呼び出します。これにより、プロンプト、生成された画像、パイプラインアーキテクチャ、および実験に関連するすべての設定が Weights & Biases に自動的にログされ、画像生成実験を再現可能で共有しやすく、分析しやすくします。
autolog(init=dict(project="sdxl"))

# ベースパイプラインを呼び出して潜在情報を生成します
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを呼び出して精錬された画像を生成します
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner,
).images[0]
```

{{% /tab %}}

{{% tab header="ノートブック" value="notebook" %}}

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

import wandb
from wandb.integration.diffusers import autolog

# SDXL ベースパイプラインを初期化します
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL リファイナーパイプラインを初期化します
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

# 実験を再現可能にするため、ランダム性をコントロールします。
# シードは自動的に WandB にログされます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers に向けて WandB Autolog を呼び出します。これにより、プロンプト、生成された画像、パイプラインアーキテクチャ、および実験に関連するすべての設定が Weights & Biases に自動的にログされ、画像生成実験を再現可能で共有しやすく、分析しやすくします。
autolog(init=dict(project="sdxl"))

# ベースパイプラインを呼び出して潜在情報を生成します
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを呼び出して精錬された画像を生成します
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner,
).images[0]

# 実験を終了します
wandb.finish()
```

{{% /tab %}}

{{< /tabpane >}}

- Stable Diffusion XL + Refiner 実験の例：
    {{< img src="/images/integrations/diffusers-autolog-6.gif" alt="オートログがどのようにして Stable Diffusion XL + Refiner 実験を追跡するかの例" >}}

## さらなるリソース

* [Stable Diffusion のためのプロンプト工学ガイド](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: Text-to-Image 生成のための拡散変圧器モデル](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)