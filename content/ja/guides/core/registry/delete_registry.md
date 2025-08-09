---
title: Registryを削除
menu:
  default:
    identifier: ja-guides-core-registry-delete_registry
    parent: registry
weight: 8
---

このページでは、Team 管理者または Registry 管理者がカスタムRegistryを削除する方法を説明します。[core registry]({{< relref path="/guides/core/registry/registry_types#core-registry" lang="ja" >}}) は削除できません。

- Team 管理者は組織内の任意のカスタムRegistryを削除できます。
- Registry 管理者は、自分が作成したカスタムRegistryを削除できます。

Registryを削除すると、そのRegistryに属するコレクションも削除されますが、Registryにリンクされたアーティファクトは削除されません。そのようなアーティファクトは、もともとログされたプロジェクト内に残ります。


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

`wandb` API の `delete()` メソッドを使って、プログラムからRegistryを削除できます。以下の例では、次の手順を紹介しています。

1. 削除したいRegistryを `api.registry()` で取得します。
1. 返ってきたRegistryオブジェクトの `delete()` メソッドを呼んで、Registryを削除します。

```python
import wandb

# W&B APIを初期化
api = wandb.Api()

# 削除したいRegistryを取得
fetched_registry = api.registry("<registry_name>")

# Registryの削除
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. 削除したいカスタムRegistryを選択します。
3. 画面右上のギアアイコンをクリックして、Registryの設定を表示します。
4. Registryを削除するには、設定ページ右上のゴミ箱アイコンをクリックしてください。
5. モーダルが表示されたら、削除するRegistry名を入力して **Delete** をクリックして確定します。

{{% /tab %}}
{{< /tabpane >}}