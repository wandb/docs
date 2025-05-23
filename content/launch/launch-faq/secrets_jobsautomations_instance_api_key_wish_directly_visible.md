---
menu:
  launch:
    identifier: secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
title: Can you specify secrets for jobs/automations? For instance, an API key which
  you do not wish to be directly visible to users?
---

Yes. Follow these steps:

1. Create a Kubernetes secret in the designated namespace for the runs using the command:  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. After creating the secret, configure the queue to inject the secret when runs start. Only cluster administrators can view the secret; end users cannot see it.