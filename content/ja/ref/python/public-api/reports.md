---
title: レポート
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B の Report オブジェクト用パブリック API。

このモジュールは、W&B Reports と連携し、レポート関連のデータを管理するためのクラスを提供します。



---

## <kbd>class</kbd> `Reports`
Reports は `BetaReport` オブジェクトのイテラブルなコレクションです。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアントインスタンス。
 - `project` (`wandb.sdk.internal.Project`):  レポートを取得する Project。
 - `name` (str, オプション):  フィルタリングするレポート名。`None` の場合はすべてのレポートを取得します。
 - `entity` (str, オプション):  プロジェクトの Entity 名。デフォルトはプロジェクトの Entity です。
 - `per_page` (int):  1ページあたりに取得するレポート数（デフォルトは 50）。


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

GraphQL のエッジを File オブジェクトへ変換します。

---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```

ページネーション用に GraphQL クエリの変数を更新します。


---

## <kbd>class</kbd> `BetaReport`
BetaReport は W&B 上で作成されたレポートに関連付けられるクラスです。

警告: この API は将来のリリースで変更される可能性があります



**属性:**
 
 - `id` (string):  レポートの一意な識別子
 - `name` (string):  レポート名
 - `display_name` (string):  レポートの表示名
 - `description` (string):  レポートの説明
 - `user` (User):  レポートを作成したユーザー（ユーザー名とメールアドレス含む）
 - `spec` (dict):  レポートの仕様
 - `url` (string):  レポートの URL
 - `updated_at` (string):  最終更新のタイムスタンプ
 - `created_at` (string):  レポート作成時のタイムスタンプ

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

レポート内の特定セクションに紐づく run を取得します。

---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

このレポートを表示する iframe を含む HTML を生成します。


---