---
title: "How does W&B Launch build images?"
tags: []
---

### How does W&B Launch build images?
The steps taken to build an image vary depending on the source of the job being run, and whether the resource configuration specifies an accelerator base image.

:::note
When specifying a queue config, or submitting a job, a base accelerator image can be provided in the queue or job resource configuration:
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

During the build process the following actions are taken dependant on the type of job and accelerator base image provided:

|                                                     | Install python using apt | Install python packages | Create a user and workdir | Copy code into image | Set entrypoint |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| Job sourced from git                                |                          |            X            |             X             |           X          |        X       |
| Job sourced from code                               |                          |            X            |             X             |           X          |        X       |
| Job sourced from git and provided accelerator image |             X            |            X            |             X             |           X          |        X       |
| Job sourced from code and provided accelerator image|             X            |            X            |             X             |           X          |        X       |
| Job sourced from image                              |                          |                         |                           |                      |                |