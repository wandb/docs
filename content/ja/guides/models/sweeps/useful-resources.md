---
title: スイープについて詳しく学ぶ
description: 役立つSweepsの情報源のコレクション。
menu:
  default:
    identifier: ja-guides-models-sweeps-useful-resources
    parent: sweeps
---

### Academic papers

Li, Lisha, et al. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

次の W&B Reports では、W&B Sweeps を使用したハイパーパラメータ最適化を探るプロジェクトの例を紹介しています。

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ベースラインの開発と Drought Watch ベンチマークへの提出を探ります。
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: 3つの異なるタスクで異なる副作用ペナルティを持つエージェントを訓練します：パターン生成、パターン削除、およびナビゲーション。
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: 信号を想像上のパターン（パレイドリア）からどのように区別しますか？この記事は、W&Bで何が可能かを示し、さらなる探索を促すことを目的としています。
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: 自然言語理解のためのモデルを探索するために Hugging Face を使用します。
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: ランダムフォレストとディープネットを使用して分子構造から化学的特性を予測します。
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: ハイパーパラメータ最適化がなぜ重要かを探り、機械学習モデルのハイパーパラメータチューニングを自動化するための3つのアルゴリズムを見てみましょう。

### selfm-anaged

次のハウツーガイドでは、W&B を使用して現実世界の問題を解決する方法を示しています：

* [Sweeps with XGBoost](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 説明: XGBoost を使用したハイパーパラメータチューニングに W&B Sweeps を使用する方法。

### Sweep GitHub repository

W&B はオープンソースを推奨し、コミュニティからの貢献を歓迎します。GitHub リポジトリは [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps) で見つけることができます。W&B オープンソース リポジトリへの貢献方法については、W&B GitHub [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md) を参照してください。