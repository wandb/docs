---
title: roc_curve()
object_type: python_sdk_custom_charts
data_type_classification: function
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

ROC 曲線（Receiver Operating Characteristic curve）チャートを作成します。



**引数:**
 
 - `y_true`:  ターゲット変数の正解クラスラベル。形状は (num_samples,) である必要があります。
 - `y_probas`:  各クラスごとの予測確率または判定スコア。形状は (num_samples, num_classes) です。
 - `labels`:  `y_true` におけるクラスインデックスに対応した人が読めるラベル。たとえば `labels=['dog', 'cat']` の場合、クラス0が'dog'、クラス1が'cat'としてプロットに表示されます。None の場合、`y_true` のインデックスがそのまま使われます（デフォルト: None）。
 - `classes_to_plot`:  ROC曲線に含めるユニークなクラスラベルのサブセット。None の場合、`y_true` に存在するすべてのクラスがプロットされます（デフォルト: None）。
 - `title`:  ROC曲線プロットのタイトル（デフォルト: "ROC Curve"）。
 - `split_table`:  テーブルを W&B UI 内で独立したセクションに分けるかどうか。`True` の場合、「Custom Chart Tables」と名付けられたセクションに表示されます（デフォルト: `False`）。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡してください。



**例外:**
 
 - `wandb.Error`:  numpy、pandas、または scikit-learn が見つからない場合に発生します。



**例:**
 ```python
import numpy as np
import wandb

# 3つの疾患での医療診断分類問題をシミュレーション
n_samples = 200
n_classes = 3

# 正解ラベル: 各サンプルに "Diabetes"、"Hypertension"、"Heart Disease" のいずれかを割り当てる
disease_labels = ["Diabetes", "Hypertension", "Heart Disease"]
# 0: Diabetes, 1: Hypertension, 2: Heart Disease
y_true = np.random.choice([0, 1, 2], size=n_samples)

# 予測確率: 各サンプルごとに合計が1となるような予測値をシミュレーション
y_probas = np.random.dirichlet(np.ones(n_classes), size=n_samples)

# プロットするクラスを指定（3疾患すべてをプロット）
classes_to_plot = [0, 1, 2]

# W&B の run を初期化し、疾患分類のための ROC 曲線チャートをログ
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