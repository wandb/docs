---
title: レポート
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-reports
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API で Report オブジェクトを操作します。

このモジュールは、W&B Reports と連携し、レポート関連のデータを管理するためのクラスを提供します。



---

## <kbd>class</kbd> `Reports`
Reports は `BetaReport` オブジェクトのイテラブルなコレクションです。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアントインスタンス
 - `project` (`wandb.sdk.internal.Project`):  レポートを取得する Project
 - `name` (str, オプション):  フィルタリング対象の Report 名。`None` の場合、すべての Report を取得
 - `entity` (str, オプション):  Project の Entity 名。指定しない場合、Project の entity が使われます。
 - `per_page` (int):  1ページあたりに取得するレポート数（デフォルトは50）

### <kbd>method</kbd> `Reports.__init__`

```python
__init__(client, project, name=None, entity=None, per_page=50)
```






---


### <kbd>property</kbd> Reports.length





---


### <kbd>method</kbd> `Reports.convert_objects`

```python
convert_objects()
```

GraphQL エッジを File オブジェクトに変換します。

---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```

ページネーション用に GraphQL クエリ変数を更新します。


---

## <kbd>class</kbd> `BetaReport`
BetaReport は W&B で作成された Report に紐づくクラスです。

警告: この API は今後のリリースで変更される可能性があります。



**属性:**
 
 - `id` (string):  レポートのユニークな識別子
 - `name` (string):  レポート名
 - `display_name` (string):  レポートの表示名
 - `description` (string):  レポートの説明
 - `user` (User):  レポートを作成したユーザー（ユーザー名とメールアドレスを含む）
 - `spec` (dict):  レポートの spec
 - `url` (string):  レポートのURL
 - `updated_at` (string):  最終更新時刻
 - `created_at` (string):  レポート作成時刻

### <kbd>method</kbd> `BetaReport.__init__`

```python
__init__(client, attrs, entity=None, project=None)
```






---

### <kbd>property</kbd> BetaReport.created_at





---

### <kbd>property</kbd> BetaReport.description





---

### <kbd>property</kbd> BetaReport.display_name





---

### <kbd>property</kbd> BetaReport.id





---

### <kbd>property</kbd> BetaReport.name





---

### <kbd>property</kbd> BetaReport.sections

レポートからパネルセクション（グループ）を取得します。

---

### <kbd>property</kbd> BetaReport.spec





---

### <kbd>property</kbd> BetaReport.updated_at





---

### <kbd>property</kbd> BetaReport.url





---

### <kbd>property</kbd> BetaReport.user







---

### <kbd>method</kbd> `BetaReport.runs`

```python
runs(section, per_page=50, only_selected=True)
```

レポートの特定セクションに紐づく Run を取得します。

---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

このレポートを表示する iframe を含む HTML を生成します。


---