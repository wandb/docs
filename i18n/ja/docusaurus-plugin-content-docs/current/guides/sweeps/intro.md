---
slug: /guides/sweeps
description: Hyperparameter search and model optimization with W&B Sweeps
---

# ハイパーパラメーターのチューニング

<head>
  <title>Tune Hyperparameters with Sweeps</title>
</head>

Weights & Biasesスウィープを使って、ハイパーパラメーター検索を自動化し、可能性があるモデルの空間を探索します。数行のコードでスウィープを作成します。スウィープは、自動化されたハイパーパラメーター検索のメリットを、可視化に富んだインタラクティブな実験トラッキングと結合します。ベイズなどの一般的な検索メソッド、グリッド検索およびランダムから選んでハイパーパラメーター空間を検索します。1台または複数台のマシンでスウィープジョブを拡大し、並列化します。


![Draw insights from large hyperparameter tuning experiments with interactive dashboards.](/images/sweeps/intro_what_it_is.png)

### 仕組み​

Weights & Biasesスウィープには2つのコンポーネント、コントローラおよび1つまたは複数のエージェントがあります。コントローラは新しいハイパーパラメーターの組み合わせを選び出します。[通常、スウィープサーバーはWeights & Biasesサーバー上で管理されます](https://docs.wandb.ai/guides/sweeps/local-controller).

エージェントはWeights & Biasesサーバーにハイパーパラメーターのクエリを行い、これらのパラメーターを使ってモデルトレーニングを実行します。その後、トレーニング結果はスウィープサーバーにレポートされます。エージェントは1つまたは複数のプロセスを、1台または複数台のマシンで実行できます。複数のプロセスを複数のマシンで実行できるエージェントの柔軟性によって、スウィープの並列化と拡大が容易になります。スウィープの拡大方法の詳細情報は、[エージェントの並列化](https://docs.wandb.ai/guides/sweeps/parallelize-agents)をご覧ください。

以下のステップに従ってW&Bスウィープを作成します：

1. **W&Bをコードに追加:** Pythonスクリプトでコード2行を追加して、ハイパーパラメーターと出力メトリクスをスクリプトから記録します。詳細情報は、[W&Bをコードに追加する](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)を参照してください。
2. **スウィープ構成を定義**: スウィープの変数と範囲を定義します。検索戦略を選びます — グリッド、ランダムおよびベイズ検索に加えて、早期終了などの反復を迅速に行うための手法もサポートしています。詳細情報は、[スウィープ構成を定義する](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。
3. **スウィープの初期化**: スウィープサーバーを起動します。弊社では、この一元化されたコントローラをホスティングし、スウィープを実行するエージェント間での調整を行います。詳細情報は、[スウィープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。
4. **スウィープの開始**: 各マシンで、スウィープでのモデルのトレーニングに使用したい1行コマンドを実行します。エージェントは集中型スウィープサーバーに次に試すハイパーパラメーターを尋ねてから、runを実行します。詳細情報は、[スウィープエージェントの開始](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)を参照してください。
5. **可視化結果（オプション**: ライブダッシュボードを開くと、すべての結果が1か所に表示されます。

### 開始方法​

ユースケースに従って、以下のリソースを探索し、Weights & Biasesスウィープの使用を開始してください:

* Weights & Biasesスウィープでハイパーパラメーターのチューニングを初めて行う場合は、[クイックスタート](https://docs.wandb.ai/guides/sweeps/quickstart)を読むことをお勧めします。クイックスタートには、最初のW&Bスウィープのセットアップ方法が説明されています。
* `Weights and Biases開発者ガイド`で、以下のようなスウィープに関するトピックを探索してください：
  * [W&Bをコードに追加する](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)
  * [スウィープ構成を定義する](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)
  * [スウィープを初期化する](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)
  * [スウィープエージェントを開始する](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)
  * [スウィープ結果を可視化する](https://docs.wandb.ai/guides/sweeps/visualize-sweep-results)
* [PyTorchでハイパーパラメータースウィープを体系化する](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing\_Hyperparameter\_Sweeps\_in\_PyTorch\_with\_W%26B.ipynb#scrollTo=e43v8-9MEoYk) をお試しください。Google Colab Jupyterノートブック JupyterノートブックでPyTorchフレームワークを使ってスウィープを作成する方法の例。
* W&Bスウィープによるハイパーパラメーターの最適化を説明する[スウィープ実験のキュレートされたリスト](https://docs.wandb.ai/guides/sweeps/useful-resources#reports-with-sweeps)をご覧ください。結果はW&Bレポートに保存されます。
* [Weights & Biases SDKリファレンスガイド](https://docs.wandb.ai/ref)をお読みください。

手順を段階的に説明した動画をご覧ください: [W&Bスウィープでハイパーパラメーターを簡単にチューニングする](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases).

<!-- {% embed url="http://wandb.me/sweeps-video" %} -->
