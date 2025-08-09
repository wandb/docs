---
title: W&B の組織からユーザーのリストをエクスポートするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-export_list_users_account
support:
- 管理者
- ユーザー管理
toc_hide: true
type: docs
url: /support/:filename
---

W&B の組織からユーザーリストをエクスポートするには、管理者が次のコードで SCIM API を利用します。

```python
import base64
import requests

# ユーザー名とAPIキーをbase64でエンコードする関数
def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # 組織の管理者ユーザー名
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

必要に応じて出力を保存するようにスクリプトを修正してください。