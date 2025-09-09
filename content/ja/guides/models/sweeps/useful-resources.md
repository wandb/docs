---
title: sweeps について詳しく学ぶ
description: Sweeps に役立つ情報源の一覧。
menu:
  default:
    identifier: ja-guides-models-sweeps-useful-resources
    parent: sweeps
---

### 学術論文

Li, Lisha, et al. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The   Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

以下の W&B Reports では、W&B Sweeps を用いたハイパーパラメータ最適化に取り組むプロジェクトの例を紹介します。

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 説明: ベースラインの構築と、Drought Watch ベンチマークへの投稿の探索。
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 説明: 異なる副作用ペナルティで学習した エージェント を、パターン生成・パターン除去・ナビゲーションの 3 つのタスクで評価します。
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 説明: シグナルとパレイドリア（見えないはずのパターンの錯覚）をどう見分けるか。この記事では W&B で実現できることを紹介し、さらなる探究を促します。
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 説明: Hugging Face を使って自然言語理解のモデルを探索します。
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 説明: 分子構造からの化学的性質を、ランダムフォレスト と ディープネット で予測します。
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 説明: ハイパーパラメータ最適化 が重要な理由を解説し、機械学習 モデル のハイパーパラメータチューニング を自動化する 3 つのアルゴリズムを紹介します。

### セルフマネージド

以下の ハウツー ガイド では、W&B を使って実世界の課題を解決する方法を紹介します。

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 説明: XGBoost を使って W&B Sweeps でハイパーパラメータチューニング を行う方法。

### Sweeps GitHub リポジトリ

W&B はオープンソースを推進しており、コミュニティからの貢献を歓迎します。[W&B Sweeps GitHub リポジトリ](https://github.com/wandb/sweeps) をご覧ください。W&B のオープンソース リポジトリへの貢献方法は、W&B GitHub の[貢献ガイドライン](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)を参照してください。