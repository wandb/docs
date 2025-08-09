---
title: Hugging Face Diffusers
menu:
  default:
    identifier: ja-guides-integrations-diffusers
    parent: integrations
weight: 120
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" >}}

[Hugging Face Diffusers](https://huggingface.co/docs/diffusers) は、画像・音声や分子の 3D 構造まで生成できる、最先端の学習済み拡散モデル用ライブラリとしてよく使われています。W&B とのインテグレーションにより、柔軟で高度な実験管理、メディアの可視化、パイプラインのアーキテクチャーや設定管理を、使いやすさを損なうことなく、対話型・集中型ダッシュボードに追加できます。

## たった 2 行で次世代のロギングが可能

実験に関連するプロンプト、ネガティブプロンプト、生成されたメディアや設定内容を、たった 2 行のコードを追加するだけで全てログできます。以下がその 2 行のコード例です：

```python
# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を実行
autolog(init=dict(project="diffusers_logging"))
```

| {{< img src="/images/integrations/diffusers-autolog-4.gif" alt="Experiment results logging" >}} | 
|:--:| 
| **実験の結果がどのようにログされるかの例。** |

## はじめに

1. `diffusers`, `transformers`, `accelerate`, `wandb` をインストールします。

    - コマンドライン:

        ```shell
        pip install --upgrade diffusers transformers accelerate wandb
        ```

    - ノートブック:

        ```bash
        !pip install --upgrade diffusers transformers accelerate wandb
        ```

2. `autolog` を使って W&B Run を初期化し、[サポートされているすべてのパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) から入出力を自動でトラッキングします。

    `autolog()` 関数は `init` パラメータを指定して呼び出すことができ、このパラメータには [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) で必要となるパラメータの辞書を渡せます。

    `autolog()` を呼び出すことで、W&B Run が初期化され、[サポートされているすべてのパイプライン呼び出し](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72) における入出力を自動的にトラッキングします。

    - 各パイプライン呼び出しごとに、ワークスペース内のそれぞれの[テーブル]({{< relref path="/guides/models/tables/" lang="ja" >}})で管理され、パイプライン呼び出しに紐づく設定は、その run のワークフロー一覧に追加されます。
    - プロンプト、ネガティブプロンプト、そして生成メディアは [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ja" >}}) に記録されます。
    - シードやパイプラインアーキテクチャーを含む、実験に紐づくその他の設定内容も、その run の config セクションに保存されます。
    - 各パイプライン呼び出しで生成されたメディアは、その run の[メディアパネル]({{< relref path="/guides/models/track/log/media" lang="ja" >}})にも表示されます。

    {{% alert %}}
    [サポートされているパイプライン呼び出しの一覧](https://github.com/wandb/wandb/blob/main/wandb/integration/diffusers/autologger.py#L12-L72)がご覧いただけます。もしこのインテグレーションに新機能のリクエストやバグ報告がある場合は、[W&B GitHub issues ページ](https://github.com/wandb/wandb/issues) にて issue を作成してください。
    {{% /alert %}}

## 実例

### Autologging

autolog が実際に動作するエンドツーエンドの簡単な例です：

{{< tabpane text=true >}}
{{% tab header="Script" value="script" %}}
```python
import torch
from diffusers import DiffusionPipeline

# autolog 関数をインポート
from wandb.integration.diffusers import autolog

# パイプラインを呼び出す前に autolog を実行
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 画像生成のためにパイプラインをコール
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

# パイプラインを呼び出す前に autolog を実行
autolog(init=dict(project="diffusers_logging"))

# 拡散パイプラインを初期化
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

# プロンプト、ネガティブプロンプト、シードを定義
prompt = ["a photograph of an astronaut riding a horse", "a photograph of a dragon"]
negative_prompt = ["ugly, deformed", "ugly, deformed"]
generator = torch.Generator(device="cpu").manual_seed(10)

# 画像生成のためにパイプラインをコール
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

- 単一実験の結果：

    {{< img src="/images/integrations/diffusers-autolog-2.gif" alt="Experiment results logging" >}}

- 複数実験の結果：

    {{< img src="/images/integrations/diffusers-autolog-1.gif" alt="Experiment results logging" >}}

- 実験の設定内容：

    {{< img src="/images/integrations/diffusers-autolog-3.gif" alt="Experiment config logging" >}}

{{% alert %}}
ノートブック（IPython など）環境下でパイプライン実行後は、必ず [`wandb.Run.finish()`]({{< relref path="/ref/python/sdk/functions/finish.md" lang="ja" >}}) を明示的に呼び出してください。Python スクリプト実行の場合は自動的に終了処理されるため不要です。
{{% /alert %}}

### 複数パイプラインによるワークフローの追跡

このセクションでは、[Stable Diffusion XL + Refiner](https://huggingface.co/docs/diffusers/using-diffusers/sdxl#base-to-refiner-model) の典型的なワークフローで autolog を使う例を紹介します。[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) で生成された latent が対応する refiner でさらに洗練されます。

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

# ランダム性を制御して実験の再現性を確保
# seed は自動で WandB にログされます
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers の WandB Autolog を呼び出し
# プロンプト、生成画像、パイプラインアーキテクチャー、全ての実験設定を
# W&B に自動でログし、実験の再現・共有・分析が容易になります
autolog(init=dict(project="sdxl"))

# ベースパイプラインを実行し latent を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを実行し最終画像を生成
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

# ランダム性を制御して実験の再現性を確保
# seed は自動で WandB にログされます
seed = 42
generator_base = torch.Generator(device="cuda").manual_seed(seed)
generator_refiner = torch.Generator(device="cuda").manual_seed(seed)

# Diffusers の WandB Autolog を呼び出し
# プロンプト、生成画像、パイプラインアーキテクチャー、全ての実験設定を
# W&B に自動でログし、実験の再現・共有・分析が容易になります
autolog(init=dict(project="sdxl"))

# ベースパイプラインを実行し latent を生成
image = base_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_type="latent",
    generator=generator_base,
).images[0]

# リファイナーパイプラインを実行し最終画像を生成
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

- Stable Diffusion XL + Refiner 実験の例：
    {{< img src="/images/integrations/diffusers-autolog-6.gif" alt="Stable Diffusion XL experiment tracking" >}}

## その他のリソース

* [Stable Diffusion のためのプロンプトエンジニアリングガイド](https://wandb.ai/geekyrakshit/diffusers-prompt-engineering/reports/A-Guide-to-Prompt-Engineering-for-Stable-Diffusion--Vmlldzo1NzY4NzQ3)
* [PIXART-α: テキストから画像生成を行う Diffusion Transformer Model](https://wandb.ai/geekyrakshit/pixart-alpha/reports/PIXART-A-Diffusion-Transformer-Model-for-Text-to-Image-Generation--Vmlldzo2MTE1NzM3)