---
title: How do I export a list of users from my W&B Organisation?
displayed_sidebar: support
tags:
- cli
- admin
---
To export a list of users from your W&B Organisation an Admin can use the SCIM API with the following code:

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # Organization admin username
key = ''  # API key
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

Ensure you have the appropriate API key and permissions. Modify scripts to save output as needed.


    