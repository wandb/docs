---
title: Jupyter ノートブックをトラッキングする
description: Jupyter と一緒に W&B を使えば、ノートブックから離れることなくインタラクティブな可視化ができます。
menu:
  default:
    identifier: ja-guides-models-track-jupyter
    parent: experiments
weight: 6
---

W&B を Jupyter で使うことで、ノートブック上でインタラクティブな可視化が可能になります。カスタム分析、実験、プロトタイプを組み合わせて、すべてを完全にログとして記録できます。

## W&B と Jupyter ノートブックの主なユースケース

1. **反復的な実験**: 実験を何度も実行し、パラメータを調整して、それぞれの run が自動的に W&B に保存されます。手動でメモを取る必要はありません。
2. **コードの保存**: モデルを再現しようとすると、どのセルが実行され、どんな順番で処理されたか分からなくなることがあります。[設定ページ]({{< relref path="/guides/models/app/settings-page/" lang="ja" >}})でコード保存を有効にすると、各実験ごとにセルの実行履歴が記録されます。
3. **カスタム分析**: run を W&B に記録した後は、API からデータフレームを取得してカスタム分析を行い、その結果も W&B にログとして残し、レポートで保存・共有できます。

## ノートブックで始めるには

以下のコードをノートブックの最初に実行し、W&B をインストールしてアカウントをリンクします。

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、実験をセットアップし、ハイパーパラメーターを保存します。

```python
wandb.init(
    project="jupyter-projo",
    config={
        "batch_size": 128,
        "learning_rate": 0.01,
        "dataset": "CIFAR-100",
    },
)
```

`wandb.init()` を実行した後、新しいセルで `%%wandb` を使うと、ノートブック内でライブグラフが表示されます。このセルを複数回実行すると、データが同じ run に追加されていきます。

```notebook
%%wandb

# ここにトレーニングループを記述
```

この [サンプルノートブック](https://wandb.me/jupyter-interact-colab) でご自身でもお試しいただけます。

{{< img src="/images/track/jupyter_widget.png" alt="Jupyter W&B ウィジェット" >}}

### ノートブックで W&B のインターフェースをライブ表示

既存のダッシュボード、sweeps、レポートも `%wandb` マジックコマンドを使って、ノートブック内に直接表示できます。

```notebook
# プロジェクトのワークスペースを表示
%wandb USERNAME/PROJECT
# 特定の run を表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# sweep を表示
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# レポートを表示
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 埋め込み iframe の高さ指定
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` や `%wandb` マジック以外にも、`wandb.init()` 実行後にセルの末尾で `wandb.Run.finish()` を呼ぶと、その場でグラフが描画されます。また、API から取得した任意のレポート、sweep、run オブジェクトに対して `ipython.display(...)` を使うこともできます。

```python
import wandb
from IPython.display import display
# run を初期化
run = wandb.init()

# セルの出力が run.finish() なら、ライブグラフが表示されます
run.finish()
```

{{% alert %}}
W&B でできることをもっと知りたい方は、[データとメディアのログ方法ガイド]({{< relref path="/guides/models/track/log/" lang="ja" >}}) や [お好きな ML ツールキットとのインテグレーション方法]({{< relref path="/guides/integrations/" lang="ja" >}})をチェック。または、[リファレンスドキュメント]({{< relref path="/ref/python/" lang="ja" >}})や [example リポジトリ](https://github.com/wandb/examples) もご覧ください。
{{% /alert %}}

## W&B の Jupyter 機能いろいろ

1. **Colab での簡単認証**: Colab で初めて `wandb.init` を呼んだとき、ブラウザで W&B にログインしていれば自動で認証されます。run ページの Overviewタブ には Colab へのリンクも表示されます。
2. **Jupyter マジック:** ダッシュボード、sweeps、レポートをノートブックで直接表示できます。`%wandb` マジックにプロジェクト・sweeps・レポートのパスを指定すると、W&B インターフェースがノートブック内にレンダリングされます。
3. **docker コンテナで Jupyter をローンチ**: `wandb docker --jupyter` を実行すると dockerコンテナ を立ち上げてコードをマウントし、Jupyter のインストールも自動で行い、ポート 8888 で起動します。
4. **セルの実行順を気にせず run を管理**: 通常、`wandb.init` が次に呼ばれた時点で run を `finished` としてマークします。これにより、データの準備・トレーニング・テスト等、どのセルをどんな順番で実行してもすべて同じ run にログされます。[設定](https://app.wandb.ai/settings)でコード保存を有効にすれば、どのセルがどんな状態で実行されたかも記録でき、複雑なパイプラインの再現も容易です。Jupyter ノートブックで run を手動で完了させる場合は `run.finish` を呼んでください。

```python
import wandb

run = wandb.init()

# ここにトレーニングスクリプトやログ出力

run.finish()
```