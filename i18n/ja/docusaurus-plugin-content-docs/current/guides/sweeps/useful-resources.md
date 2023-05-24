---
description: Collection of useful sources for Sweeps.
displayed_sidebar: default
---

# 便利なリソース

<head>
  <title>W&Bスイープについてもっと学ぶためのリソース</title>
</head>

### 学術論文

Li, Lishaら。"[Hyperband: ハイパーパラメータ最適化への新しいバンディットベースのアプローチ](https://arxiv.org/pdf/1603.06560.pdf)" _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816。

### スイープ実験

以下のW&Bレポートでは、W&Bスイープを用いたハイパーパラメータ最適化を探求するプロジェクトの事例を紹介しています。

* [ドラウトウォッチベンチマークの進捗](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ドラウトウォッチベンチマークへの提出を探求するためのベースラインの開発。
* [強化学習における安全性ペナルティの調整](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: 異なる副作用ペナルティを用いて訓練されたエージェントを、パターン作成、パターン除去、ナビゲーションの3つの異なるタスクで調査します。
* [W&Bとハイパーパラメータ探索の意味とノイズ](https://wandb.ai/stacey/pytorch\_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: どのようにして信号と幻覚パターン（想像上のパターン）を区別するのか？この記事では、W&Bで可能なことを紹介し、さらなる探求にインスピレーションを与えることを目指しています。
* [誰が彼ら？トランスフォーマーによるテキスト曖昧性解消](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: Hugging Faceを使って自然言語理解のためのモデルを探索
* [DeepChem: 分子の溶解性](https://wandb.ai/stacey/deepchem\_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: ランダムフォレストとディープネットワークを用いて、分子構造から化学的性質を予測。
* [MLOps入門: ハイパーパラメーターチューニング](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: ハイパーパラメータ最適化がなぜ重要であるのか、機械学習モデルのハイパーパラメータチューニングを自動化するための3つのアルゴリズムを紹介します。
### ハウツーガイド

以下のハウツーガイドでは、Weights & Biasesを使って実際の問題解決方法を紹介しています:

* [Sweeps with XGBoost](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)

  * 説明: XGBoostを用いたハイパーパラメータチューニングにW&B Sweepsを使用する方法です。

### Sweep GitHubリポジトリ

Weights & Biasesはオープンソースであり、コミュニティからの貢献を歓迎しています。GitHubリポジトリは[https://github.com/wandb/sweeps](https://github.com/wandb/sweeps)で見つけることができます。Weights & Biasesのオープンソースリポジトリへの貢献方法については、W&B GitHubの[Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)を参照してください。