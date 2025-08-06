---
title: スイープについて詳しく知る
description: スイープに関する便利な情報源のまとめ。
menu:
  default:
    identifier: useful-resources
    parent: sweeps
---

### 論文

Li, Lisha, ほか. 「[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)」 _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

以下の W&B Reports では、W&B Sweeps を用いたハイパーパラメータ最適化を探求するプロジェクト例を紹介しています。

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ベースラインの開発や Drought Watch ベンチマークへの提出についての検討。
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: 3つの異なるタスク（パターン作成・削除・ナビゲーション）に対し、異なる副作用ペナルティで訓練したエージェントを検証しています。
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: 本記事では W&B の機能紹介と、さらなる探究へのインスピレーションを目的とし、どのようにシグナルと疑似パターン（空想上のパターン）を区別するかを扱います。
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: Hugging Face を使って自然言語理解のためのモデルを検証します。
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: ランダムフォレストとディープネットを用いた、分子構造から化学的性質を予測する方法。
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: なぜハイパーパラメータ最適化が重要なのかを解説し、機械学習モデルのハイパーパラメータチューニングを自動化する3つのアルゴリズムを紹介します。

### selfm-anaged

以下のハウツーガイドでは、W&B を使った現実的な課題の解決方法を解説しています。

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 説明: XGBoost を使ったハイパーパラメータチューニングへの W&B Sweeps の適用方法。

### Sweep GitHub repository

W&B はオープンソースを推進しており、コミュニティからのコントリビューションを歓迎しています。 [W&B Sweeps GitHub repository](https://github.com/wandb/sweeps) をご覧ください。W&B オープンソースリポジトリへの貢献方法については、W&B GitHub の [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md) をご確認ください。