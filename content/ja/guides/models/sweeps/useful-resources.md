---
title: スイープの詳細について詳しく見る
description: スイープに役立つ情報源のまとめ。
menu:
  default:
    identifier: ja-guides-models-sweeps-useful-resources
    parent: sweeps
---

### 学術論文

Li, Lisha, 他. 「[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)」 _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

以下の W&B Reports では、W&B Sweeps を活用したハイパーパラメータ最適化の実例を紹介しています。

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ベースラインの構築や、Drought Watch ベンチマークへのサブミッションを幅広く検討しています。
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: 三つの異なるタスク（パターン作成、パターン除去、ナビゲーション）で、異なる副作用ペナルティによるエージェントの訓練結果を比較します。
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: シグナルと擬似的なパターン（pareidolia）をどのように区別するか？本記事は W&B を使った可能性を幅広く紹介し、さらなる探究を後押しします。
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: Hugging Face を活用し、自然言語理解向けのモデルを探究しています。
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: ランダムフォレストやディープネットを用いて、分子構造から化学的性質を予測します。
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: ハイパーパラメータ最適化が重要な理由や、自動チューニングのための三つのアルゴリズムを、機械学習モデルを例に解説します。

### Self-managed

以下のハウツーガイドでは、W&B を使って実際の課題をどのように解決できるかを紹介しています：

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 説明: XGBoost を用いて、W&B Sweeps でハイパーパラメータチューニングを行う方法。

### Sweep GitHub リポジトリ

W&B はオープンソースを推進しており、コミュニティからの貢献を歓迎しています。[W&B Sweeps GitHub リポジトリ](https://github.com/wandb/sweeps) をご覧ください。W&B オープンソースリポジトリへの貢献方法については、W&B GitHub の [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md) を参照してください。