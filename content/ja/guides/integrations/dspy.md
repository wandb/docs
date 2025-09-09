---
title: DSPy
description: W&B で DSPy プログラムを追跡し、最適化する。
menu:
  default:
    identifier: ja-guides-integrations-dspy
    parent: integrations
weight: 80
---

W&B と DSPy を併用して、言語モデル プログラムをトラッキングし、最適化しましょう。W&B は [Weave DSPy integration](https://weave-docs.wandb.ai/guides/integrations/dspy) を補完し、次を提供します:
- 評価メトリクスの経時トラッキング
- プログラム シグネチャの変化を可視化する W&B Tables
- MIPROv2 のような DSPy オプティマイザーとのインテグレーション

DSPy モジュールの最適化を行う際に包括的な可観測性を得るには、W&B と Weave の両方でインテグレーションを有効にしてください。

{{< alert title="注意" color="info" >}}
`wandb==0.21.2` および `weave==0.52.5` 時点では、Weave は W&B と併用すると自動で初期化されます:

- `weave` を import してから `wandb.init()` を呼び出した場合（スクリプトの場合）
- 先に `wandb.init()` を呼び、その後で `weave` を import した場合（ノートブック / Jupyter の場合）

明示的な `weave.init(...)` の呼び出しは不要です。
{{< /alert >}}

## インストールと認証

必要なライブラリをインストールし、W&B で認証します:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" %}}

1. 必要なライブラリをインストールします:

    ```shell
    pip install wandb weave dspy
    ```

1. `WANDB_API_KEY` の [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を設定し、ログインします:

    ```bash
    export WANDB_API_KEY=<your_api_key>
    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" %}}
1. 必要なライブラリをインストールします:

    ```bash
    pip install wandb weave dspy
    ```
1. コード内で W&B にログインします:

    ```python
    import wandb
    wandb.login()
    ```
{{% /tab %}}

{{% tab header="ノートブック" %}}
必要なライブラリをインストールして import し、W&B にログインします:
```notebook
!pip install wandb weave dspy

import wandb
wandb.login()
```
{{% /tab %}}
{{< /tabpane >}}

W&B を初めてお使いですか？[クイックスタート ガイド]({{< relref path="/guides/quickstart.md" lang="ja" >}}) をご覧ください。


## プログラムの最適化をトラッキング（実験的） {#track-program-optimization}

MIPROv2 など、`dspy.Evaluate` を使用する DSPy のオプティマイザーでは、`WandbDSPyCallback` を使って評価メトリクスを時系列でログし、W&B Tables でプログラム シグネチャの変化を追跡できます。

```python
import dspy
from dspy.datasets import MATH

import weave
import wandb
from wandb.integration.dspy import WandbDSPyCallback

# W&B を初期化（weave を import するだけで十分。明示的な weave.init は不要）
project_name = "dspy-optimization"
wandb.init(project=project_name)

# DSPy に W&B コールバックを追加
dspy.settings.callbacks.append(WandbDSPyCallback())

# 言語モデルを設定
teacher_lm = dspy.LM('openai/gpt-4o', max_tokens=2000, cache=True)
student_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
dspy.configure(lm=student_lm)

# データセットを読み込み、プログラムを定義
dataset = MATH(subset='algebra')
program = dspy.ChainOfThought("question -> answer")

# オプティマイザーを設定して実行
optimizer = dspy.MIPROv2(
    metric=dataset.metric,
    auto="light",
    num_threads=24,
    teacher_settings=dict(lm=teacher_lm),
    prompt_model=student_lm
)

optimized_program = optimizer.compile(
    program,
    trainset=dataset.train,
    max_bootstrapped_demos=2,
    max_labeled_demos=2
)
```

このコードを実行すると、W&B Run の URL と Weave の URL の両方が得られます。W&B では、評価メトリクスの推移に加え、プログラム シグネチャの変化を示す Tables が表示されます。Run の **Overview** タブには、詳細確認のための Weave のトレースへのリンクが含まれています。

{{< img src="/images/integrations/dspy_run_page.png" alt="W&B における DSPy 最適化の Run" >}}

Weave によるトレース、DSPy を用いた評価と最適化の詳細は、[Weave DSPy インテグレーション ガイド](https://weave-docs.wandb.ai/guides/integrations/dspy) を参照してください。

## 予測を W&B Tables にログする

最適化中に個々のサンプルを確認できるよう、詳細な予測ログ記録を有効にします。このコールバックは評価ステップごとに W&B Tables を作成し、成功と失敗の要因分析に役立ちます。

```python
from wandb.integration.dspy import WandbDSPyCallback

# 予測のロギングを有効化（デフォルトで有効）
callback = WandbDSPyCallback(log_results=True)
dspy.settings.callbacks.append(callback)

# 最適化を実行
optimized_program = optimizer.compile(program, trainset=train_data)

# 必要に応じて予測のロギングを無効化
# callback = WandbDSPyCallback(log_results=False)
```

### 予測データへのアクセス

最適化後、W&B で予測データを確認します:

1. Run の **Overview** ページに移動します。
2. `predictions_0`、`predictions_1` のようなパターン名の Table パネルを探します。
3. `is_correct` でフィルタして失敗例を分析します。
4. Project Workspace で Runs 間のテーブルを比較します.

各テーブルには次の列が含まれます:
- `example`: 入力データ
- `prediction`: モデル出力
- `is_correct`: 評価結果

詳しくは、[W&B Tables ガイド](../models/tables/visualize-tables.md) と [Tables チュートリアル](../../tutorials/tables.md) を参照してください。

## DSPy プログラムを保存してバージョン管理する

優れた DSPy プログラムを再現しバージョン管理できるよう、W&B Artifacts として保存します。プログラム全体を保存するか、状態のみを保存するかを選べます。

```python
from wandb.integration.dspy import WandbDSPyCallback

# コールバックのインスタンスを作成
callback = WandbDSPyCallback()
dspy.settings.callbacks.append(callback)

# 最適化を実行
optimized_program = optimizer.compile(program, trainset=train_data)

# 保存オプション:

# 1. プログラム全体（推奨） - アーキテクチャーと状態を含む
callback.log_best_model(optimized_program, save_program=True)

# 2. 状態のみを JSON で保存 - 軽量で人が読みやすい
callback.log_best_model(optimized_program, save_program=False, filetype="json")

# 3. 状態のみを pickle で保存 - Python オブジェクトを保持
callback.log_best_model(optimized_program, save_program=False, filetype="pkl")

# バージョン管理のためにカスタム エイリアスを追加
callback.log_best_model(
    optimized_program,
    save_program=True,
    aliases=["best", "production", "v2.0"]
)
```