---
title: Reports
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-reports
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/reports.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B の Report オブジェクト向けの公開 API。 

このモジュールは、W&B の レポート とやり取りし、レポート関連のデータを管理するためのクラスを提供します。 



---

## <kbd>class</kbd> `Reports`
Reports は `BetaReport` オブジェクトの遅延イテレータです。 



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用する API クライアントインスタンス。 
 - `project` (`wandb.sdk.internal.Project`):  レポートを取得する対象のプロジェクト。 
 - `name` (str, optional):  フィルタに使うレポート名。`None` の場合はすべてのレポートを取得。 
 - `entity` (str, optional):  プロジェクトの entity 名。指定しない場合はプロジェクトの entity が既定。 
 - `per_page` (int):  1 ページあたりに取得するレポート数（既定は 50）。 

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

GraphQL のエッジを File オブジェクトに変換します。 

---

### <kbd>method</kbd> `Reports.update_variables`

```python
update_variables()
```

ページネーションのために GraphQL のクエリ変数を更新します。 


---

## <kbd>class</kbd> `BetaReport`
BetaReport は、W&B で作成されたレポートに対応するクラスです。 

レポートの属性（name、description、user、spec、タイムスタンプ）へのアクセスや、関連する run やセクションの取得、レポートを HTML としてレンダリングするためのメソッドを提供します。 



**属性:**
 
 - `id` (string):  レポートの一意な識別子。 
 - `display_name` (string):  人間が読みやすいレポートの表示名。 
 - `name` (string):  レポート名。よりユーザーにわかりやすい名前には `display_name` を使用してください。 
 - `description` (string):  レポートの説明。 
 - `user` (User):  このレポートを作成したユーザーの情報（ユーザー名、メールアドレス）を含む辞書。 
 - `spec` (dict):  レポートの spec。 
 - `url` (string):  レポートの URL。 
 - `updated_at` (string):  最終更新時刻のタイムスタンプ。 
 - `created_at` (string):  レポートが作成された時刻のタイムスタンプ。 

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

レポートからパネルのセクション（グループ）を取得します。 

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

レポートのセクションに関連付けられた run を取得します。 

---

### <kbd>method</kbd> `BetaReport.to_html`

```python
to_html(height=1024, hidden=False)
```

このレポートを表示する iframe を含む HTML を生成します。 


---