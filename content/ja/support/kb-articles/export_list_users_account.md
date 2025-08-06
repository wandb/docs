---
title: W&B オーガニゼーションからユーザー一覧をエクスポートするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 管理者
- ユーザー管理
---

W&B の組織からユーザー一覧をエクスポートするには、管理者が SCIM API を以下のコードで利用します。

```python
import base64
import requests

def encode_base64(username, key):
    # 認証用の文字列を Base64 でエンコードする
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # 組織管理者のユーザー名
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
    # 各ユーザーの userName とメールアドレスをリストに追加
    users.append([user['userName'], user['emails']['Value']])
```

必要に応じて、出力結果を保存するようにスクリプトを修正してください。