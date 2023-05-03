# ログ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1555-L1750)

現在のrunの履歴にデータの辞書をログします。

```python
log(
 data: Dict[str, Any],
 step: Optional[int] = None,
 commit: Optional[bool] = None,
 sync: Optional[bool] = None
) -> None
```

`wandb.log`を使って、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなどのrunsのデータをログします。

ライブ例、コードスニペット、ベストプラクティスなどについては、[ログのガイド](https://docs.wandb.ai/guides/track/log)を参照してください。

最も基本的な使い方は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})` です。これにより、損失と精度がrunの履歴に保存され、これらのメトリクスのサマリー値が更新されます。
[wandb.ai](https://wandb.ai)のワークスペースでログされたデータを可視化するか、W&Bアプリの[セルフホストインスタンス](https://docs.wandb.ai/guides/hosting)でローカルに可視化するか、たとえばJupyterノートブックでローカルに可視化・探索するためにデータをエクスポートして、[弊社のAPI](https://docs.wandb.ai/guides/track/public-api-guide)を使用します。

UIでは、サマリー値がrunテーブルに表示され、run間の単一値を比較できます。
サマリー値は、`wandb.run.summary["key"] = value`で直接設定することもできます。

ログされる値はスカラーである必要はありません。wandbオブジェクトのログがサポートされています。
例えば、`wandb.log({"example": wandb.Image("myimage.jpg")})`は、W&B UIで適切に表示されるサンプル画像をログします。
サポートされているすべてのタイプについては、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types)を参照していただくか、3D分子構造やセグメンテーションマスク、PR曲線やヒストグラムの例がある[ログガイド](https://docs.wandb.ai/guides/track/log)を参照してください。
`wandb.Table`は、構造化データのログに使用できます。詳細については、[テーブルのログガイド](https://docs.wandb.ai/guides/data-vis/log-tables)を参照してください。

W&B UIでは、入れ子になったメトリクスのログが推奨され、サポートされています。
`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`のような入れ子の辞書でログすると、W&B UIでは、メトリクスが`train`および`val`セクションに整理されます。

wandbはグローバルステップを追跡し、デフォルトでは`wandb.log`の呼び出し毎にインクリメントされます。そのため、関連するメトリクスを一緒にログすることが推奨されます。関連するメトリクスを一緒にログするのが不便な場合は、`wandb.log({"train-loss": 0.5}, commit=False)`を呼び出してから`wandb.log({"accuracy": 0.9})`を呼び出すと、`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`を呼び出すのと同等です。
`wandb.log`は、1秒あたり数回以上呼び出すことを想定していません。
もっと頻繁にログを取りたい場合は、クライアント側でデータを集約するか、
パフォーマンスが低下する可能性があります。

| 引数 | |
| :--- | :--- |
| `data` | (辞書型, 任意) シリアライズ可能なPythonオブジェクト、つまり `str`、`ints`、`floats`、`Tensors`、`dicts`、または `wandb.data_types` のいずれか。 |
| `commit` | (ブール値, 任意) メトリクス辞書をwandbサーバーに保存し、ステップをインクリメントします。Falseの場合、`wandb.log`はデータ引数とメトリクスを現在のメトリクス辞書に更新し、`wandb.log`が`commit=True`で呼び出されるまでメトリクスは保存されません。 |
| `step` | (整数, 任意) 処理のグローバルステップ。これにより、コミットされていない以前のステップが保持されますが、指定されたステップはデフォルトでコミットされません。 |
| `sync` | (ブール値, True) この引数は廃止予定であり、現在 `wandb.log` の振る舞いは変わりません。 |



#### 例:

より多く、より詳細な例については、
[ログの取り方に関するガイド](https://docs.wandb.com/guides/track/log) を参照してください。

### 基本的な使い方

```python
import wandb

wandb.init()
wandb.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルログ

```python
import wandb
wandb.init()
wandb.log({"loss": 0.2}, commit=False)
# このステップを報告する準備ができたときに別の場所で:
wandb.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムに勾配をサンプリング
gradients = np.random.randn(100, 100)
wandb.init()
wandb.log({"gradients": wandb.Histogram(gradients)})
```

### numpyからの画像

```python
import numpy as np
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
 image = wandb.Image(pixels, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```
以下は、Markdownのテキストを翻訳してください。日本語に翻訳して、それ以外のことは言わないでください。テキスト：

### PILからの画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
 pil_image = PILImage.fromarray(pixels, mode="RGB")
 image = wandb.Image(pil_image, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```

### numpyからのビデオ

```python
import numpy as np
import wandb

wandb.init()
# 軸は (時間, チャンネル, 高さ, 幅)です
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```
### Matplotlibプロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y) # y = x^2 のプロット
wandb.log({"chart": fig})
```

### PR曲線
```python
wandb.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
```

### 3Dオブジェクト
```python
wandb.log(
 {
 "generated_samples": [
 wandb.Object3D(open("sample.obj")),
 wandb.Object3D(open("sample.gltf")),
 wandb.Object3D(open("sample.glb")),
 ]
 }
)
```
| Raises | |

| :--- | :--- |

| `wandb.Error` | `wandb.init`の前に呼び出された場合 |

| `ValueError` | 無効なデータが渡された場合 |