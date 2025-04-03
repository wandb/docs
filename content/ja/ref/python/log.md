---
title: log
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

`log` を使用して、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなど、run からデータをログに記録します。

ライブの例、コードスニペット、ベストプラクティスなどについては、[ログ記録に関するガイド](https://docs.wandb.ai/guides/track/log) を参照してください。

最も基本的な使い方は、`run.log({"train-loss": 0.5, "accuracy": 0.9})` です。
これにより、損失と精度がrun の履歴に保存され、これらのメトリクスの概要値が更新されます。

[wandb.ai](https://wandb.ai) のワークスペース、または W&B アプリの[セルフホストインスタンス](https://docs.wandb.ai/guides/hosting) でログに記録されたデータを視覚化するか、[API](https://docs.wandb.ai/guides/track/public-api-guide) を使用してデータをエクスポートし、Jupyter ノートブックなどのローカルで視覚化および探索します。

ログに記録される値は、スカラーである必要はありません。wandb オブジェクトのログ記録はすべてサポートされています。
たとえば、`run.log({"example": wandb.Image("myimage.jpg")})` は、W&B UI に適切に表示されるサンプル画像をログに記録します。
サポートされているすべての異なるタイプについては、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types) を参照するか、3D 分子構造やセグメンテーションマスクから PR 曲線やヒストグラムまで、例については[ログ記録に関するガイド](https://docs.wandb.ai/guides/track/log) を確認してください。
`wandb.Table` を使用して構造化されたデータをログに記録できます。詳細については、[テーブルのログ記録に関するガイド](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) を参照してください。

W&B UI は、名前にフォワードスラッシュ（`/`）が付いたメトリクスを、最後のスラッシュの前のテキストを使用して名前が付けられたセクションに整理します。たとえば、以下は「train」と「validate」という名前の2つのセクションになります。

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

ネストできるレベルは1つのみです。`run.log({"a/b/c": 1})` は、「a/b」という名前のセクションを生成します。

`run.log` は、1秒あたり数回以上呼び出されることは想定されていません。
最適なパフォーマンスを得るには、ログ記録を N 回のイテレーションごとに1回に制限するか、複数のイテレーションでデータを収集して、1つのステップでログに記録します。

### W&B ステップ

基本的な使用法では、`log` を呼び出すたびに新しい「ステップ」が作成されます。
ステップは常に増加する必要があり、前のステップにログを記録することはできません。

任意のメトリクスをグラフの X 軸として使用できることに注意してください。
多くの場合、W&B ステップをトレーニングステップとしてではなく、タイムスタンプのように扱う方が適しています。

```
# 例: X 軸として使用する「epoch」メトリクスをログに記録します。
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/run#define_metric) も参照してください。

複数の `log` 呼び出しを使用して、`step` および `commit` パラメータを使用して同じステップにログを記録できます。
以下はすべて同等です。

```
# 通常の用法:
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
|  `data` |  `str` キーと、`int`、`float`、`string` などのシリアル化可能な Python オブジェクトである値を持つ `dict`。`wandb.data_types`、シリアル化可能な Python オブジェクトのリスト、タプル、NumPy 配列。この構造の他の `dict`。 |
|  `step` |  ログに記録するステップ番号。`None` の場合、暗黙的な自動インクリメントステップが使用されます。説明の注記を参照してください。 |
|  `commit` |  true の場合、ステップを確定してアップロードします。false の場合、ステップのデータを累積します。説明の注記を参照してください。`step` が `None` の場合、デフォルトは `commit=True` です。それ以外の場合、デフォルトは `commit=False` です。 |
|  `sync` |  この引数は非推奨であり、何も行いません。 |

#### 例:

より詳細な例については、[ログ記録に関するガイド](https://docs.wandb.com/guides/track/log) を参照してください。

### 基本的な使用法

```python
import wandb

with wandb.init() as run:
    run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルログ記録

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # このステップを報告する準備ができたら、別の場所で:
    run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムに勾配をサンプリング
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy からの画像

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

### PIL からの画像

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

### numpy からの動画

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 軸は (時間、チャンネル、高さ、幅)
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
    ax.plot(x, y)  # y = x^2 をプロット
    run.log({"chart": fig})
```

### PR 曲線

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

| 発生 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` の前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |
```