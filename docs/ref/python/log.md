
# log

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1665-L1877' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

現在の run の履歴にデータの辞書をログします。

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

`wandb.log` を使用して run からデータをログします。例えばスカラー、画像、ビデオ、
ヒストグラム、プロット、テーブルなどです。

ライブ例、コードスニペット、ベストプラクティスなどは [logging ガイド](https://docs.wandb.ai/guides/track/log) を参照してください。

最も基本的な使い方は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})` です。
これは損失と精度を run の履歴に保存し、これらのメトリクスの要約値を更新します。

workspace でログされたデータを [wandb.ai](https://wandb.ai) で可視化するか、
W&B アプリの [セルフホスティングインスタンス](https://docs.wandb.ai/guides/hosting)
でローカルに可視化するか、データをエクスポートして Jupyter ノートブックなどでローカルに
可視化および探索できます。[API ガイド](https://docs.wandb.ai/guides/track/public-api-guide) も参照してください。

UI では、要約値が run テーブルに表示され、run 間で単一の値を比較します。
要約値は `wandb.run.summary["key"] = value` を使用して直接設定することもできます。

ログされた値はスカラーである必要はありません。任意の wandb オブジェクトをログすることがサポートされています。
例えば `wandb.log({"example": wandb.Image("myimage.jpg")})` は例として画像をログし、
W&B UI にうまく表示されます。
サポートされているすべての種類については、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types) を参照するか、
3D 分子構造やセグメンテーションマスクから PR 曲線やヒストグラムまでの例については [logging ガイド](https://docs.wandb.ai/guides/track/log) を確認してください。
`wandb.Table` を使用して構造化データをログすることもできます。[テーブルログのガイド](https://docs.wandb.ai/guides/data-vis/log-tables) を参照してください。

ネストされたメトリクスのログが推奨されており、W&B UI でサポートされています。
例えば、`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})` というネストされた辞書でログすると、
メトリクスは W&B UI に `train` と `val` のセクションに整理されます。

wandb はグローバルステップを追跡しており、デフォルトでは `wandb.log` を呼び出すたびに増加します。そのため、関連するメトリクスを一緒にログすることが推奨されます。
もし関連するメトリクスを一緒にログするのが不便な場合には、
`wandb.log({"train-loss": 0.5}, commit=False)` と呼び出し、その後
`wandb.log({"accuracy": 0.9})` と呼び出すことで、
`wandb.log({"train-loss": 0.5, "accuracy": 0.9})` と同等になります。

`wandb.log` は一秒間に数回以上呼び出すことを意図していません。
それ以上頻繁にログしたい場合は、クライアント側でデータを集約する方がよく、パフォーマンスが低下する可能性があります。

| 引数 |  |
| :--- | :--- |
|  `data` |  (辞書, オプション) シリアライズ可能な Python オブジェクトの辞書、例えば `str`, `ints`, `floats`, `Tensors`, `dicts`, または任意の `wandb.data_types`. |
|  `commit` |  (ブール, オプション) メトリクス辞書を wandb サーバーに保存し、ステップをインクリメントします。false の場合、`wandb.log` は現在のメトリクス辞書を引数のデータで更新するだけで、`commit=True` で呼ばれるまでメトリクスは保存されません。 |
|  `step` |  (整数, オプション) グローバルなプロセッシングステップ。この引数は未コミットの以前のステップを保持しますが、指定されたステップをコミットしないのがデフォルトです。 |
|  `sync` |  (ブール, True) この引数は廃止され、現在は `wandb.log` の動作を変更しません。 |

#### 例:

さらに詳細な例については、
[logging ガイド](https://docs.wandb.com/guides/track/log) を参照してください。

### 基本的な使い方

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルログ

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 別の場所でこのステップを報告する準備ができたら:
run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムにサンプリングした勾配
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy から画像をログ

```python
import numpy as np
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = wandb.Image(pixels, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### PIL から画像をログ

```python
import numpy as np
from PIL import Image as PILImage
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### numpy からビデオをログ

```python
import numpy as np
import wandb

run = wandb.init()
# 軸は (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib プロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # プロット y = x^2
run.log({"chart": fig})
```

### PR カーブ

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D オブジェクト

```python
import wandb

run = wandb.init()
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
|  `wandb.Error` |  `wandb.init` の前に呼び出されると発生します |
|  `ValueError` |  無効なデータが渡されると発生します |