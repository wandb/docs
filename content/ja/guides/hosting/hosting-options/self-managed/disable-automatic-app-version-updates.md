---
title: W&B サーバー の自動更新を無効にする
description: W&B サーバーの自動更新を無効化する方法を説明します。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-disable-automatic-app-version-updates
    parent: self-managed
weight: 99
---

W&B Server の自動バージョンアップグレードを無効にし、そのバージョンを固定する方法を示します。これらの手順は、[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) によって管理されているデプロイメントにのみ適用されます。

{{% alert %}}
W&B は、W&B Server のメジャーリリースを最初のリリース日から 12 ヶ月間サポートします。**Self-managed** インスタンスをご利用のお客様は、サポートを維持するために期限内にアップグレードする責任があります。サポートされていないバージョンを使い続けないでください。W&B は、**Self-managed** インスタンスをご利用のお客様に対し、サポートを維持し、最新の機能、パフォーマンスの向上、および修正を受け取るために、少なくとも四半期に一度はデプロイメントを最新リリースに更新することを強く推奨します。
{{% alert %}}

## 要件

- W&B Kubernetes Operator `v1.13.0` 以降
- System Console `v2.12.2` 以降

これらの要件を満たしていることを確認するには、インスタンスの W&B Custom Resource または Helm チャートを参照してください。`operator-wandb` および `system-console` コンポーネントの `version` の値を確認してください。

## 自動アップデートの無効化
1. `admin` ロールを持つユーザーとして W&B App にログインします。
2. 上部にあるユーザーアイコンをクリックし、次に **System Console** をクリックします。
3. **Settings** > **Advanced** に移動し、**Other** タブを選択します。
4. **Disable Auto Upgrades** セクションで、**Pin specific version** をオンにします。
5. **Select a version** ドロップダウンをクリックし、W&B Server のバージョンを選択します。
6. **Save** をクリックします。

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="自動アップデート保存済み" >}}

    自動アップグレードはオフになり、W&B Server は選択したバージョンに固定されます。
7. 自動アップグレードがオフになっていることを確認します。**Operator** タブに移動し、reconciliation ログで文字列 `Version Pinning is enabled` を検索します。

```
│info 2025-04-17T17:24:16Z wandb default No changes found
│info 2025-04-17T17:24:16Z wandb default Active spec found
│info 2025-04-17T17:24:16Z wandb default Desired spec
│info 2025-04-17T17:24:16Z wandb default License
│info 2025-04-17T17:24:16Z wandb default Version Pinning is enabled
│info 2025-04-17T17:24:16Z wandb default Found Weights & Biases instance, processing the spec...
│info 2025-04-17T17:24:16Z wandb default === Reconciling Weights & Biases instance...
```