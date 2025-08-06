---
title: Hugging Face Diffusers
menu:
  default:
    identifier: diffusers
    parent: integrations
weight: 120
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" >}}

[Hugging Face Diffusers](https://huggingface.co/docs/diffusers) は、画像・音声・分子の 3D 構造まで生成できる、最先端の学習済みディフュージョンモデル用ライブラリです。W&B のインテグレーションを使うことで、実験管理やメディア可視化、パイプラインアーキテクチャーや設定管理など、インタラクティブな集中型ダッシュボードに柔軟でリッチな機能を追加できます。それでいて、簡単な使い勝手はそのままです。

## たった２行で次世代ログ

プロンプトやネガティブプロンプト、生成されたメディア、実験に関する設定などを、たった２行のコードでまとめてログできます。以下が、その２行のコードです。

```python
# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼ぶ
autolog(init=dict(project="diffusers_logging"))
```

| {{< img src="/images/integrations/diffusers-autolog-4.gif" alt="Experiment results logging" >}} | 
|:--:| 
| **実験の結果がどのようにログされるかの例です。** |

## はじめよう

1. `diffusers`、`transformers`、`accelerate`、`wandb` をインストールします。

    - コマンドラインの場合:

        ```shell
        pip install --upgrade diffusers transformers accelerate wandb
        ```

    - ノートブックの場合:

        ```bash
        !pip install --upgrade diffusers transformers accelerate wandb
        ```


2. `autolog` を使って W&B Run を初期化し、[全てのサポートされているパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) の入力と出力を自動でトラッキングします。

    `autolog()` に `init` 引数を指定できます。これは [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) に必要なパラメータの辞書を渡します。

    `autolog()` を呼び出すと、W&B Run が初期化され、[全てのサポートされているパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) の入力・出力が自動でトラッキングされます。

    - 各パイプライン呼び出しごとに [table]({{< relref "/guides/models/tables/" >}}) がワークスペースに作成され、そのパイプライン呼び出しに紐付く設定（config）は run のワークフローリストに追加されます。
    - プロンプト、ネガティブプロンプト、生成されたメディアは [`wandb.Table`]({{< relref "/guides/models/tables/" >}}) にも記録されます。
    - シード値やパイプラインアーキテクチャーなど、その他実験に関する設定は、その run の config セクションに保存されます。
    - 各パイプライン呼び出しで生成されたメディアは、run の [media panels]({{< relref "/guides/models/track/log/media" >}}) にも記録されます。

    {{% alert %}}
    [サポートされているパイプライン呼び出し一覧](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)を見ることができます。もし新しい機能追加のリクエストやバグ報告があれば、[W&B の GitHub issues ページ](https://github.com/wandb/wandb/issues) で issue を作成してください。
    {{% /alert %}}

## 実例

### オートロギング

autolog の動作例を、エンドツーエンドで簡単にご紹介します。

{{< tabpane text=true >}}
{{% tab header="Script" value="script" %}}
```python
import torch
from diffusers import DiffusionPipeline

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を呼ぶ
autolog(init=dict(project="diffusers_logging"))

# ディフュージョンパイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
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
{{% /tab %}}

{{% tab header="Notebook" value="notebook"%}}
```python
import torch
from diffusers import DiffusionPipeline

import wandb

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

run = wandb.init()

# パイプラインを呼び出す前に autolog を呼ぶ
autolog(init=dict(project="diffusers_logging"))

# ディフュージョンパイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
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
run.finish()
```
{{% /tab %}}
{{< /tabpane >}}


- 1つの実験の例:

    {{< img src="/images/integrations/diffusers-autolog-2.gif" alt="Experiment results logging" >}}

- 複数実験の例:

    {{< img src="/images/integrations/diffusers-autolog-1.gif" alt="Experiment results logging" >}}

- 実験設定（config）の例:

    {{< img src="/images/integrations/diffusers-autolog-3.gif" alt="Experiment config logging" >}}

{{% alert %}}
パイプラインを呼び出した後、IPython ノートブック環境で実行する場合は、必ず [`wandb.Run.finish()`]({{< relref "/ref/python/sdk/functions/finish.md" >}}) を明示的に呼び出してください。Python スクリプトではこの必要はありません。
{{% /alert %}}

### マルチパイプラインワークフローのトラッキング

このセクションでは、[Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) ワークフローで autolog を使う例を紹介します。`StableDiffusionXLPipeline` で生成した潜在変数（latents）を、対応するリファイナーでさらに洗練させます。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb" >}}

{{< tabpane text=true >}}

{{% tab header="Python Script" value="script" %}}

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

# 実験の再現性のため、ランダム値を制御しています。
# シード値も自動で WandB にログされます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers 用の WandB Autolog を呼びます。自動で
# プロンプトや生成画像、パイプラインアーキテクチャー、設定など
# 実験に必要な情報を W&B に記録し、再現・共有・分析しやすくします。
autolog(init=dict(project="sdxl"))

# ベースパイプラインで潜在変数を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインで画像を洗練
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

# 実験の再現性のため、ランダム値を制御しています。
# シード値も自動で WandB にログされます。
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers 用の WandB Autolog を呼びます。自動で
# プロンプトや生成画像、パイプラインアーキテクチャー、設定など
# 実験に必要な情報を W&B に記録し、再現・共有・分析しやすくします。
autolog(init=dict(project="sdxl"))

# ベースパイプラインで潜在変数を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインで画像を洗練
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

- Stable Diffusion XL + Refiner 実験の例:
    {{< img src="/images/integrations/diffusers-autolog-6.gif" alt="Stable Diffusion XL experiment tracking" >}}

## その他のリソース

* [A Guide to Prompt Engineering for Stable Diffusion](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: A Diffusion Transformer Model for Text-to-Image Generation](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)