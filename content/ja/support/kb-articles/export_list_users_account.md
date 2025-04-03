---
title: How do I export a list of users from my W&B Organisation?
menu:
  support:
    identifier: ja-support-kb-articles-export_list_users_account
support:
- administrator
- user management
toc_hide: true
type: docs
url: /support/:filename
---

W&B の組織から ユーザー のリストをエクスポートするには、管理者は次の コード で SCIM API を使用します。

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # Organization admin username
key = ''  # APIキー
scim_base_url = 'https://api.wandb.ai/scim/v2'
users_endpoint = f'{scim_base_url}/Users'
headers = {
    'Authorization': f'Basic {encode_base64(username, key)}',
    'Content-Type': 'application/scim+json'
}

response = requests.get(users_endpoint, headers=headers)
users = []
for user in response.json()['Resources']:
    users.append([user['userName'], user['emails']['Value']])
```

必要に応じて、出力結果を保存するように スクリプト を変更してください。
