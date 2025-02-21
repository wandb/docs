---
title: Learn more about sweeps
description: Sweeps に役立つ情報源を集めました。
menu:
  default:
    identifier: ja-guides-models-sweeps-useful-resources
    parent: sweeps
---

### 学術論文

Li, Lisha, 他. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

以下の W&B の Reports は、W&B の Sweeps を使用したハイパーパラメーター最適化を調査する project の例を示しています。

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ベースラインを開発し、Drought Watch ベンチマークへの提出を検討します。
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: パターン作成、パターン削除、ナビゲーションの 3 つの異なるタスクで、異なる副作用ペナルティでトレーニングされた agent を検証します。
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: シグナルとパレイドリア (想像上のパターン) をどのように区別しますか?この記事では、W&B で何が可能かを紹介し、さらなる探求を促すことを目的としています。
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: 自然言語理解のためのモデルを調査するために Hugging Face を使用します
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: ランダムフォレストと深層ネットを使用して、分子構造から化学的特性を予測します。
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: ハイパーパラメーター最適化が重要な理由を探り、機械学習モデルのハイパーパラメーターチューニングを自動化するための 3 つのアルゴリズムを見てみましょう。

### selfm-anaged

次のハウツー ガイドでは、W&B で実際の問題を解決する方法を示します。

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 説明: XGBoost を使用したハイパーパラメーターチューニングに W&B Sweeps を使用する方法。

### Sweep GitHub リポジトリ

W&B はオープンソースを提唱し、コミュニティからの貢献を歓迎します。GitHub リポジトリは [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps) にあります。W&B オープンソースリポジトリへの貢献方法については、W&B GitHub の [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md) を参照してください。
