---
title: Hugging Face Diffusers
menu:
  default:
    identifier: ja-guides-integrations-diffusers
    parent: integrations
weight: 120
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" >}}

[Hugging Face Diffusers](https://huggingface.co/docs/diffusers) は、画像・音声、さらには分子の 3D 構造まで生成できる最先端の 学習済み 拡散モデルの定番 ライブラリ です。W&B インテグレーションは、その使いやすさを損なうことなく、柔軟でリッチな 実験管理、メディアの可視化、パイプライン アーキテクチャー、設定 管理を、インタラクティブで一元化された ダッシュボード に追加します。

## たった 2 行でワンランク上のロギング

2 行の コード を含めるだけで、実験に紐づくプロンプト、ネガティブ プロンプト、生成メディア、config をすべて ログ できます。以下がロギングを始めるための 2 行です:

```python
# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼ぶ前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))
```

| {{< img src="/images/integrations/diffusers-autolog-4.gif" alt="実験結果のロギング" >}} | 
|:--:| 
| **実験結果がどのようにロギングされるかの例。** |

## はじめに

1. `diffusers`、`transformers`、`accelerate`、`wandb` をインストールします。

    - コマンドライン:

        ```shell
        pip install --upgrade diffusers transformers accelerate wandb
        ```

    - ノートブック:

        ```bash
        !pip install --upgrade diffusers transformers accelerate wandb
        ```


2. `autolog` を使って W&B Run を初期化し、[サポートされているすべてのパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)の入力と出力を自動で追跡します。

    `autolog()` 関数は `init` パラメータを受け取り、これは [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に必要なパラメータの辞書を渡せます。

    `autolog()` を呼ぶと、W&B Run が初期化され、[サポートされているすべてのパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)の入力と出力が自動で追跡されます。

    - 各パイプライン呼び出しは Workspace 内の専用の [テーブル]({{< relref path="/guides/models/tables/" lang="ja" >}}) に記録され、その呼び出しに紐づく config は、その Run の config にあるワークフロー一覧へ追加されます。
    - プロンプト、ネガティブ プロンプト、生成メディアは [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) にログされます。
    - シードやパイプライン アーキテクチャーを含む、実験に関連するその他の config は、その Run の config セクションに保存されます。
    - 各パイプライン呼び出しで生成されたメディアは、その Run の [メディア パネル]({{< relref path="/guides/models/track/log/media" lang="ja" >}}) にもログされます。

    {{% alert %}}
    [サポートされているパイプライン呼び出しの一覧](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)を参照できます。このインテグレーションへの新機能の要望やバグ報告がある場合は、[W&B GitHub issues ページ](https://github.com/wandb/wandb/issues)で Issue を作成してください。
    {{% /alert %}}

## 例

### 自動ロギング

autolog の動作を示す、簡単なエンドツーエンドの例です:

{{< tabpane text=true >}}
{{% tab header="スクリプト" value="script" %}}
```python
import torch
from diffusers import DiffusionPipeline

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼ぶ前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブ プロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 画像を生成するためにパイプラインを呼び出す
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

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

run = wandb.init()

# パイプラインを呼ぶ前に autolog を呼び出す
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブ プロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 画像を生成するためにパイプラインを呼び出す
images = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=2,
    generator=generator,
)

# 実験を終了
run.finish()
```
{{% /tab %}}
{{< /tabpane >}}


- 単一の実験の結果:

    {{< img src="/images/integrations/diffusers-autolog-2.gif" alt="実験結果のロギング" >}}

- 複数の実験の結果:

    {{< img src="/images/integrations/diffusers-autolog-1.gif" alt="実験結果のロギング" >}}

- 実験の config:

    {{< img src="/images/integrations/diffusers-autolog-3.gif" alt="実験の config のロギング" >}}

{{% alert %}}
パイプライン呼び出し後に IPython ノートブック環境でコードを実行する場合は、明示的に [`wandb.Run.finish()`]({{< relref path="/ref/python/sdk/functions/finish.md" lang="ja" >}}) を呼び出す必要があります。Python スクリプトを実行する場合は不要です。
{{% /alert %}}

### 複数パイプラインのワークフローを追跡

このセクションでは、典型的な [Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) のワークフローを autolog でデモします。ここでは、[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) が生成した潜在表現を対応する Refiner で精緻化します。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb" >}}

{{< tabpane text=true >}}

{{% tab header="Python スクリプト" value="script" %}}

```python
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

# SDXL のベース パイプラインを初期化
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL のリファイナー パイプラインを初期化
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

# 乱数を制御して実験の再現性を確保
# シードは自動的に W&B にログされます
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers 用の W&B Autolog を呼び出します。
# プロンプト、生成画像、パイプライン アーキテクチャー、
# 関連する実験の config が自動で W&B にログされ、
# 画像生成の実験を簡単に再現・共有・分析できるようになります。
autolog(init=dict(project="sdxl"))

# ベース パイプラインを呼び出して潜在表現を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナー パイプラインを呼び出して画像を精緻化
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

run = wandb.init()

# SDXL のベース パイプラインを初期化
base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipeline.enable_model_cpu_offload()

# SDXL のリファイナー パイプラインを初期化
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

# 乱数を制御して実験の再現性を確保
# シードは自動的に W&B にログされます
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers 用の W&B Autolog を呼び出します。
# プロンプト、生成画像、パイプライン アーキテクチャー、
# 関連する実験の config が自動で W&B にログされ、
# 画像生成の実験を簡単に再現・共有・分析できるようになります。
autolog(init=dict(project="sdxl"))

# ベース パイプラインを呼び出して潜在表現を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナー パイプラインを呼び出して画像を精緻化
image = refiner_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image[None, :],
    generator=generator_refiner,
).images[0]

# 実験を終了
run.finish()
```

{{% /tab %}}

{{< /tabpane >}}

- Stable Diffusion XL + Refiner の実験例:
    {{< img src="/images/integrations/diffusers-autolog-6.gif" alt="Stable Diffusion XL の実験管理" >}}

## 参考リソース

* [Stable Diffusion のためのプロンプト エンジニアリング ガイド](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: Text-to-Image 生成のための Diffusion Transformer モデル](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)