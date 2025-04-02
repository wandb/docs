---
title: watch
menu:
  reference:
    identifier: ja-ref-python-watch
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2801-L2836 >}}

指定されたPyTorch model (s) に フック して、勾配とモデルの計算グラフを監視します。

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

この関数は、トレーニング中に パラメータ 、 勾配 、またはその両方を追跡できます。将来的には、任意の 機械学習 モデルをサポートするように拡張する必要があります。

| Args |  |
| :--- | :--- |
|  models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): 監視対象の単一のモデルまたはモデルのシーケンス。 criterion (Optional[torch.F]): 最適化される損失関数 (オプション)。 log (Optional[Literal["gradients", "parameters", "all"]]): 「勾配」、「 パラメータ 」、または「すべて」を ログ に記録するかどうかを指定します。 ログ を無効にするには、None に設定します (デフォルト="gradients")。 log_freq (int): 勾配と パラメータ を ログ に記録する頻度 (バッチ単位) (デフォルト=1000)。 idx (Optional[int]): `wandb.watch` で複数のモデルを追跡する場合に使用されるインデックス (デフォルト=None)。 log_graph (bool): モデルの計算グラフを ログ に記録するかどうか (デフォルト=False)。 |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init` が呼び出されていない場合、またはモデルのいずれかが `torch.nn.Module` のインスタンスでない場合。 |
