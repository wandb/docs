---
title: roc_curve()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-roc_curve
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/roc_curve.py >}}




### <kbd>function</kbd> `roc_curve`

```python
roc_curve(
    y_true: 'Sequence[numbers.Number]',
    y_probas: 'Sequence[Sequence[float]] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[numbers.Number] | None' = None,
    title: 'str' = 'ROC Curve',
    split_table: 'bool' = False
) → CustomChart
```

受信者動作特性（ROC）曲線チャートを作成します。



**引数:**
 
 - `y_true`:  ターゲット変数の正解クラスラベル。形状は (num_samples,) です。
 - `y_probas`:  各クラスに対する予測確率または決定スコア。形状は (num_samples, num_classes) です。
 - `labels`:  `y_true` 内のクラスインデックスに対応する読みやすいラベル。例えば、`labels=['dog', 'cat']` の場合、クラス 0 は 'dog'、クラス 1 は 'cat' としてプロット内で表示されます。None の場合は `y_true` の生のクラスインデックスが使用されます。デフォルトは None です。
 - `classes_to_plot`:  ROC 曲線に含めるユニークなクラスラベルのサブセット。None の場合、`y_true` の全クラスがプロットされます。デフォルトは None です。
 - `title`:  ROC 曲線プロットのタイトル。デフォルトは "ROC Curve" です。
 - `split_table`:  テーブルを W&B UI 上で別セクションとして分割表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションでテーブルが表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡します。



**例外:**
 
 - `wandb.Error`:  numpy, pandas, または scikit-learn が見つからない場合に発生します。



**例:**
 ```python
import numpy as np
import wandb

# 3つの疾患に対する医療診断分類問題をシミュレーションします
n_samples = 200
n_classes = 3

# 正解ラベル: 各サンプルに「Diabetes」「Hypertension」「Heart Disease」を割り当てる
disease_labels = ["Diabetes", "Hypertension", "Heart Disease"]
# 0: Diabetes, 1: Hypertension, 2: Heart Disease
y_true = np.random.choice([0, 1, 2], size=n_samples)

# 予測確率: 各サンプルで合計が 1 になるような予測値をシミュレーション
y_probas = np.random.dirichlet(np.ones(n_classes), size=n_samples)

# プロットするクラスを指定（3疾患すべてをプロット）
classes_to_plot = [0, 1, 2]

# W&B の run を初期化し、疾患分類用のROC曲線プロットをログする
with wandb.init(project="medical_diagnosis") as run:
    roc_plot = wandb.plot.roc_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=disease_labels,
         classes_to_plot=classes_to_plot,
         title="ROC Curve for Disease Classification",
    )
    run.log({"roc-curve": roc_plot})
```