---
title: "How does W&B Launch build images?"
tags:
   - launch
---
The steps for building an image depend on the job source and the specified accelerator base image in the resource configuration.

:::note
When configuring a queue or submitting a job, include a base accelerator image in the queue or job resource configuration:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

The build process includes the following actions based on the job type and provided accelerator base image:

|                                                     | Install Python using apt | Install Python packages | Create a user and workdir | Copy code into image | Set entrypoint |
|