---
title: How do I export a list of users from my W&B Organisation?
toc_hide: true
type: docs
tags:
- administrator
- user management
---
To export a list of users from a W&B organization, an admin uses the SCIM API with the following code:

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

Modify the script to save the output as needed.