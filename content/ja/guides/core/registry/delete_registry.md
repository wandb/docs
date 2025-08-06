---
title: レジストリを削除
menu:
  default:
    identifier: delete_registry
    parent: registry
weight: 8
---

このページでは、Team admin または Registry admin がカスタムレジストリを削除する方法を説明します。[core registry]({{< relref "/guides/core/registry/registry_types#core-registry" >}}) は削除できません。

- Team admin は、その組織内の任意のカスタムレジストリを削除できます。
- Registry admin は、自分が作成したカスタムレジストリを削除できます。

レジストリを削除すると、そのレジストリに属するコレクションも削除されますが、レジストリに紐づく Artifacts は削除されません。そのような Artifact は、元々ログされたプロジェクト内に残ります。

{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

`wandb` API の `delete()` メソッドを使って、プログラム上でレジストリを削除できます。以下の例では、次の操作方法を示します。

1. `api.registry()` を使って削除したいレジストリを取得します。
1. 取得したレジストリオブジェクトで `delete()` メソッドを呼び出し、レジストリを削除します。

```python
import wandb

# W&B API を初期化
api = wandb.Api()

# 削除したいレジストリを取得
fetched_registry = api.registry("<registry_name>")

# レジストリを削除
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. https://wandb.ai/registry/ の **Registry** App にアクセスします。
2. 削除したいカスタムレジストリを選択します。
3. 右上の歯車アイコンをクリックし、レジストリの設定を表示します。
4. 設定ページ右上のゴミ箱アイコンをクリックしてレジストリを削除します。
5. 表示されたモーダルで削除するレジストリの名前を入力し、**Delete** をクリックして確定します。

{{% /tab %}}
{{< /tabpane >}}