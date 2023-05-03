---
slug: /guides/sweeps
description: W&B スイープを使ったハイパーパラメータ探索とモデルの最適化
---

# ハイパーパラメータをチューニングする

<head>
  <title>スイープでハイパーパラメータをチューニングする</title>
</head>

Weights & Biases スイープを使って、ハイパーパラメータ探索を自動化し、可能なモデルの空間を調べます。コード数行でスイープを作成できます。スイープは、自動化されたハイパーパラメータ探索の利点を、視覚化豊富でインタラクティブな実験追跡と組み合わせます。ベイズ、グリッドサーチ、ランダムなどの人気のある探索方法から選択して、ハイパーパラメータ空間を探索できます。スイープジョブを1台以上のマシンにスケーリングおよび並列化します。

![インタラクティブなダッシュボードを使った大規模なハイパーパラメータチューニング実験からの洞察](/images/sweeps/intro_what_it_is.png)

### 仕組み

Weights & Biases スイープには、_コントローラー_ と 1 つ以上の _エージェント_ の 2 つのコンポーネントがあります。コントローラは、新しいハイパーパラメータの組み合わせを選びます。[通常、スイープサーバーは Weights & Biases サーバーで管理されます](https://docs.wandb.ai/guides/sweeps/local-controller)。

エージェントは、Weights & Biases サーバーからハイパーパラメーターを問い合わせ、それらを使用してモデルのトレーニングを実行します。トレーニングの結果は、スイープサーバーに報告されます。エージェントは、1台以上のマシンで1つ以上のプロセスを実行できます。エージェントが複数のプロセスを複数のマシンで実行できる柔軟性で、スイープを並列化およびスケーリングしやすくなります。スイープのスケーリング方法についての詳細は、[エージェントの並列化](https://docs.wandb.ai/guides/sweeps/parallelize-agents)を参照してください。

以下の手順で W&B スイープを作成します。

1. **コードに W&B を追加する:** Python スクリプトに、ハイパーパラメータと出力メトリクスをログに記録するためのコードを数行追加します。詳細については、[コードに W&B を追加する](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)を参照してください。
2. **スイープ構成の定義:** スイープ対象となる変数と範囲を定義します。検索戦略を選択します。グリッド、ランダム、ベイズ探索などがサポートされており、早期停止などの高速化技術も利用できます。詳細については、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。
3. **スイープの初期化**: スイープサーバーを開始します。当社では、この中央コントローラをホストし、スイープを実行するエージェント間で調整を行います。詳細については、[スイープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。
4. **スイープを開始する:** スイープ内でモデルをトレーニングしたい各マシンで、1行のコマンドを実行します。エージェントは、次に試すハイパーパラメータを中央のスイープサーバーに尋ね、実行を実行します。詳細については、[スイープエージェントの開始](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)を参照してください。
5. **結果の可視化（任意）**: ライブダッシュボードを開き、すべての結果を1つの中央の場所に表示します。

### 使い始める方法
ユースケースに応じて、以下のリソースを参照して、Weights & Biases Sweepsを始めてください。



* Weights & Biases Sweepsで初めてハイパーパラメータチューニングを行う場合は、[Quickstart](https://docs.wandb.ai/guides/sweeps/quickstart)をお読みいただくことをお勧めします。クイックスタートでは、最初のW&Bスイープの設定方法を説明しています。

* Weights and Biases Developer Guideで、Sweepsに関する以下のトピックを探索してください:

  * [コードにW&Bを追加する](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)

  * [スイープ構成を定義する](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)

  * [スイープを初期化する](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)

  * [スイープエージェントを開始する](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)

  * [スイープ結果を可視化する](https://docs.wandb.ai/guides/sweeps/visualize-sweep-results)

* PyTorchフレームワークを用いたJupyterノートブックでスイープを作成する方法の例として、[Organizing Hyperparameter Sweeps in PyTorch](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing\_Hyperparameter\_Sweeps\_in\_PyTorch\_with\_W%26B.ipynb#scrollTo=e43v8-9MEoYk) Google Colab Jupyterノートブックを試してください。

* W&B Sweepsを用いたハイパーパラメータ最適化を探る[厳選されたスイープ実験のリスト](https://docs.wandb.ai/guides/sweeps/useful-resources#reports-with-sweeps) を探索してください。結果はW&B Reportsに保存されます。

* [Weights & Biases SDK リファレンスガイド](https://docs.wandb.ai/ref)をお読みください。



ステップバイステップのビデオはこちらをご覧ください：[W&Bスイープを使って簡単にハイパーパラメータをチューニングする](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases)。



<!-- {% embed url="http://wandb.me/sweeps-video" %} -->