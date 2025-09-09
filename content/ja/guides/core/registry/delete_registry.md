---
title: レジストリの削除
menu:
  default:
    identifier: ja-guides-core-registry-delete_registry
    parent: registry
weight: 8
---

このページでは、Team admin または Registry admin がカスタムレジストリを削除する方法を示します。[コアレジストリ]({{< relref path="/guides/core/registry/registry_types#core-registry" lang="ja" >}}) は削除できません。

- Team admin は、組織内の任意のカスタムレジストリを削除できます。
- Registry admin は、自身が作成したカスタムレジストリを削除できます。

レジストリを削除すると、そのレジストリに属するコレクションも削除されますが、レジストリにリンクされている Artifacts は削除されません。それらの Artifacts は、ログに記録された元の Projects に残ります。


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

`wandb` API の `delete()` メソッドを使用して、プログラムでレジストリを削除します。次の例は、その方法を示しています。

1. `api.registry()` を使用して、削除したいレジストリをフェッチします。
2. 返されたレジストリオブジェクトで `delete()` メソッドを呼び出して、レジストリを削除します。

```python
import wandb

# W&B API を初期化します
api = wandb.Api()

# 削除したいレジストリをフェッチします
fetched_registry = api.registry("<registry_name>")

# レジストリを削除します
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. https://wandb.ai/registry/ の **Registry** App に移動します。
2. 削除したいカスタムレジストリを選択します。
3. 右上隅にある歯車アイコンをクリックして、レジストリの設定を表示します。
4. レジストリを削除するには、設定ページの右上隅にあるゴミ箱アイコンをクリックします。
5. 表示されるモーダルにレジストリの名前を入力して削除するレジストリを確認し、**Delete** をクリックします。

{{% /tab %}}
{{< /tabpane >}}