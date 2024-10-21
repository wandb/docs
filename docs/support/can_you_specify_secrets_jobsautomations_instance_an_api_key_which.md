---
title: "Can you specify secrets for jobs/automations? For instance, an API key which you do not wish to be directly visible to users?"
tags:
   - launch
---

Yes. The suggested way is:

  1. Add the secret as a vanilla k8s secret in the namespace where the runs will be created. something like `kubectl create secret -n <namespace> generic <secret_name> <secret value>`

 2. Once that secret is created, you can specify a queue config to inject the secret when runs start. The end users cannot see the secret, only cluster admins can.