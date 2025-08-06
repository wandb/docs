---
title: レジストリを削除
menu:
  default:
    identifier: ja-guides-core-registry-delete_registry
    parent: registry
weight: 8
---

このページでは、Team 管理者または Registry 管理者がカスタムレジストリを削除する方法を説明します。[core registry]({{< relref path="/guides/core/registry/registry_types#core-registry" lang="ja" >}}) は削除できません。

- Team 管理者は組織内の任意のカスタムレジストリを削除できます。
- Registry 管理者は、自分が作成したカスタムレジストリを削除できます。

レジストリを削除すると、そのレジストリに属するコレクションも削除されますが、レジストリにリンクされたアーティファクトは削除されません。そのようなアーティファクトは、もともとログされたプロジェクト内に残ります。


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

`wandb` API の `delete()` メソッドを使って、プログラムからレジストリを削除できます。以下の例では、次の手順を紹介しています。

1. 削除したいレジストリを `api.registry()` で取得します。
1. 返ってきたレジストリオブジェクトの `delete()` メソッドを呼んで、レジストリを削除します。

```python
import wandb

# W&B APIを初期化
api = wandb.Api()

# 削除したいレジストリを取得
fetched_registry = api.registry("<registry_name>")

# レジストリの削除
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. 削除したいカスタムレジストリを選択します。
3. 画面右上のギアアイコンをクリックして、レジストリの設定を表示します。
4. レジストリを削除するには、設定ページ右上のゴミ箱アイコンをクリックしてください。
5. モーダルが表示されたら、削除するレジストリ名を入力して **Delete** をクリックして確定します。

{{% /tab %}}
{{< /tabpane >}}