---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registryはプライベートプレビュー中です。早期アクセスについては、アカウントチームまたはsupport@wandb.comにお問い合わせください。  
:::

W&B Registryを使用して、機械学習パイプラインからのアーティファクト（機械学習やデータセットなど）を組織内のチーム間で共有します。

## 仕組み
W&B Registryは、レジストリ、コレクション、および[artifact versions](../artifacts/create-a-new-artifact-version.md)の3つの主要なコンポーネントで構成されています。

*レジストリ*は、同種のML資産のリポジトリーやカタログです。レジストリをディレクトリの最上位レベルと考えることができます。各レジストリはコレクションと呼ばれる1つ以上のサブディレクトリで構成されています。*コレクション*は、レジストリ内のリンクされた[artifact versions](../artifacts/create-a-new-artifact-version.md)のフォルダーまたはセットです。[*artifact version*](../artifacts/create-a-new-artifact-version.md)は、特定の開発段階におけるアーティファクトの単一で不変なスナップショットです。レジストリは特定のチームではなく、組織に属します。

![](/images/registry/registry_diagram_homepage.png)

次の例を考えてみましょう。あなたの会社が組織内で新しい猫分類モデルを探索したいとします。このために、「Cat Models registry」と呼ばれるレジストリを作成します。どの画像分類が最適か分からないため、異なるハイパーパラメーターとアルゴリズムを用いて複数の機械学習実験を行います。実験と結果を整理するために、各モデルアルゴリズムに対してコレクションを作成します。そのコレクション内で、実験からの最良のモデルアーティファクトをそのコレクションにリンクします。

## W&B Model RegistryからW&B Registryへの移行

チームが現在W&B Model Registryを使用してモデルを整理している場合、新しいRegistryアプリUIを通じて引き続き使用できます。ホームページからModel Registryに移動し、バナーでチームを選択してそのモデルレジストリを訪問することができます。

![](/images/registry/nav_to_old_model_reg.gif)

現行のモデルレジストリの内容を新しいモデルレジストリに移行するためのマイグレーションに関する情報をお待ちください。移行に関する質問や懸念がある場合は、support@wandb.comにお問い合わせいただくか、製品チームにご相談ください。