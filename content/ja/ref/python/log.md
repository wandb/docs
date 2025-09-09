---
title: ログ
menu:
  reference:
    identifier: ja-ref-python-log
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1613-L1873 >}}

run データをアップロードします。

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None,
    sync: (bool | None) = None
) -> None
```

`log` を使用して、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなど、run からデータをログします。

ライブ例、コードスニペット、ベストプラクティスなどについては、[ロギングのガイド](https://docs.wandb.ai/guides/track/log)をご覧ください。

最も基本的な使用法は `run.log({"train-loss": 0.5, "accuracy": 0.9})` です。これにより、損失と精度が run の履歴に保存され、これらのメトリクスの要約値が更新されます。

[wandb.ai](https://wandb.ai) のワークスペースでログデータを視覚化するか、W&B アプリの[セルフホストインスタンス](https://docs.wandb.ai/guides/hosting)でローカルに視覚化するか、または [API](https://docs.wandb.ai/guides/track/public-api-guide) を使用してローカルでデータをエクスポートして視覚化および探索します。

ログされた値はスカラーである必要はありません。任意の wandb オブジェクトのログがサポートされています。たとえば、`run.log({"example": wandb.Image("myimage.jpg")})` は例の画像をログし、W&B UI で美しく表示されます。サポートされるすべての異なるタイプについては、[参照ドキュメント](https://docs.wandb.ai/ref/python/sdk/data-types/)または [ロギングのガイド](https://docs.wandb.ai/guides/track/log)をチェックしてみてください。3D 分子構造やセグメンテーションマスクから PR 曲線やヒストグラムまでの例を見ることができます。構造化データをログするには `wandb.Table` を使用できます。詳細は[テーブルのロギングガイド](https://docs.wandb.ai/guides/models/tables/tables-walkthrough)を参照してください。

W&B UI は、名前にフォワードスラッシュ (`/`) が含まれるメトリクスを、最後のスラッシュの前のテキストを使用して名前付けされたセクションに整理します。たとえば、次の例では、「train」と「validate」という2つのセクションが作成されます：

```
run.log(
    {
        "train/accuracy": 0.9,
        "train/loss": 30,
        "validate/accuracy": 0.8,
        "validate/loss": 20,
    }
)
```

ネストは1レベルのみサポートされています。`run.log({"a/b/c": 1})` は「a/b」という名前のセクションを生成します。

`run.log` は、1 秒間に数回以上呼び出されることを意図していません。最適なパフォーマンスのために、ログを N 回の反復ごとに 1 回に制限するか、複数の反復にわたってデータを収集し、単一のステップでログを行うようにしてください。

### W&B ステップ

基本的な使用法では、`log` を呼び出すたびに新しい「ステップ」が作成されます。ステップは常に増加しなければならず、以前のステップにログすることはできません。

チャートで任意のメトリックを X 軸として使用できることに注意してください。多くの場合、W&B ステップをタイムスタンプではなくトレーニングステップのように扱った方が良い場合があります。

```
# 例: X 軸として使用するために "epoch" メトリックをログします。
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/sdk/run/#method-rundefine_metric) も参照してください。

`step` と `commit` パラメータを使用して、同じステップにログするために複数の `log` 呼び出しを使用することができます。以下の例はすべて同等です：

```
# 通常の使用法:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 自動インクリメントなしの暗黙的なステップ:
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# 明示的なステップ:
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| 引数 |  |
| :--- | :--- |
|  `data` |  `str` キーと直列化可能な Python オブジェクトを含む `dict`。これには、`int`、`float`、`string`、任意の `wandb.data_types`、直列化可能な Python オブジェクトのリスト、タプル、NumPy 配列、同じ構造の他の `dict` が含まれます。 |
|  `step` |  ログするステップ番号。`None` の場合、暗黙的な自動インクリメントステップが使用されます。説明の中の注釈を参照してください。 |
|  `commit` |  true の場合、ステップを確定してアップロードします。false の場合は、ステップのデータを蓄積します。説明の中の注釈を参照してください。`step` が `None` の場合、デフォルトは `commit=True` です。それ以外の場合、デフォルトは `commit=False` です。 |
|  `sync` |  この引数は廃止されており、何もしません。 |

#### 例：

より多くの詳細な例については、[ロギングのガイド](https://docs.wandb.com/guides/track/log)を参照してください。

### 基本的な使用法

```python
import wandb

with wandb.init() as run:
    run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルロギング

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 別の場所で、このステップを報告する準備ができたとき：
    run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムに勾配をサンプリングします
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
```

### NumPy の画像

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### PIL の画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
        pixels = np.random.randint(
            low=0,
            high=256,
            size=(100, 100, 3),
            dtype=np.uint8,
        )
        pil_image = PILImage.fromarray(pixels, mode="RGB")
        image = wandb.Image(pil_image, caption=f"random field {i}")
        examples.append(image)
    run.log({"examples": examples})
```

### NumPy のビデオ

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 軸は (time, channel, height, width)
    frames = np.random.randint(
        low=0,
        high=256,
        size=(10, 3, 100, 100),
        dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib プロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # プロット y = x^2
    run.log({"chart": fig})
```

### PR カーブ

```python
import wandb

with wandb.init() as run:
    run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D オブジェクト

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "generated_samples": [
                wandb.Object3D(open("sample.obj")),
                wandb.Object3D(open("sample.gltf")),
                wandb.Object3D(open("sample.glb")),
            ]
        }
    )
```

| 例外 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` の前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |