# log

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1665-L1877' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

現在の run の履歴にデータの辞書を記録します。

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

`wandb.log`を使って、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなどのデータをrunから記録します。

ライブ例、コードスニペット、ベストプラクティス、その他については、[guides to logging](https://docs.wandb.ai/guides/track/log)をご覧ください。

最も基本的な使い方は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})`です。これにより、損失率と精度がrunの履歴に保存され、これらのメトリクスの要約値が更新されます。

記録されたデータは[wandb.ai](https://wandb.ai)のワークスペースで、またはW&Bアプリの[自己ホスト型インスタンス](https://docs.wandb.ai/guides/hosting)上でローカルに可視化およびエクスポートし、例えばJupyterノートブックで[API](https://docs.wandb.ai/guides/track/public-api-guide)を使ってローカルで可視化および探索することができます。

UIでは、要約値がrunテーブルに表示され、run間の個別の値を比較できます。要約値は、`wandb.run.summary["key"] = value`を使って直接設定することも可能です。

記録される値はスカラーである必要はありません。任意のwandbオブジェクトを記録することがサポートされています。例えば、`wandb.log({"example": wandb.Image("myimage.jpg")})`は例として画像を記録し、W&B UIに美しく表示されます。サポートされているすべての種類については[reference documentation](https://docs.wandb.com/ref/python/data-types)を、または3D分子構造やセグメンテーションマスクからPRカーブやヒストグラムまでの[guides to logging](https://docs.wandb.ai/guides/track/log)を参照してください。`wandb.Table`sを使用して構造化データを記録することもできます。詳細については[guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)をご覧ください。

入れ子のメトリクスをログに記録することが推奨されており、W&B UIでサポートされています。例えば、`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`のように入れ子の辞書でログを記録すると、メトリクスはW&B UIで`train`と`val`のセクションに整理されます。

wandb はグローバルステップを追跡し、デフォルトでは`wandb.log`が呼び出されるたびにステップが増加します。そのため、関連するメトリクスを一緒にログに記録することが推奨されます。関連するメトリクスを一緒にログに記録するのが不便な場合、`wandb.log({"train-loss": 0.5}, commit=False)`を呼び、次に`wandb.log({"accuracy": 0.9})`を呼ぶのは、`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`を呼ぶのと同等です。

`wandb.log`は1秒につき数回以上呼び出すことを意図していません。それより頻繁にログを記録したい場合は、クライアント側でデータを集約する方がパフォーマンスの低下を避けることができます。

| 引数 |  |
| :--- | :--- |
|  `data` |  (辞書, オプション) シリアライズ可能なPythonオブジェクト、つまり`str`、`ints`、`floats`、`Tensors`、`dicts`、または任意の`wandb.data_types`の辞書。 |
|  `commit` |  (ブール値, オプション) メトリクスの辞書をwandbサーバーに保存し、ステップを増加させます。falseの場合、`wandb.log`は現在のメトリクスの辞書を引数のデータで更新するだけで、`commit=True`を指定して`wandb.log`が呼ばれるまではメトリクスは保存されません。 |
|  `step` |  (整数, オプション) プロセッシング中のグローバルステップ。このステップは任意で以前の未コミットのステップを永続化します。 |
|  `sync` |  (ブール値, 真) この引数は非推奨であり、現在`wandb.log`の動作を変更しません。 |

#### 例:

より多く、そして詳細な例については、[our guides to logging](https://docs.wandb.com/guides/track/log)を参照してください。

### 基本的な使い方

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルロギング

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

# 正規分布からランダムにサンプルした勾配
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpyからの画像

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

### PILからの画像

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

### numpyからのビデオ

```python
import numpy as np
import wandb

run = wandb.init()
# 軸は (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlibプロット

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

### PRカーブ

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3Dオブジェクト

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
|  `wandb.Error` |  `wandb.init`の前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |