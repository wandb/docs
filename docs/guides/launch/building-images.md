---
description: Discover how to create a job for W&B Launch.
displayed_sidebar: default
---

# Building Images

When providing a job sourced from a git source or code artifact source, several steps are taken by W&B Launch to build the resulting container. Launch uses provided requirements.txt files, or if not resolves the environment from the packages used from the run that created the job.

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

During the build process the following actions are taken dependant on the type of job and accelerator base image provided:

|                                                     | Install python using apt | Install python packages | Create a user and workdir | Copy code into image | Set entrypoint |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| Job sourced from git                                |                          |            X            |             X             |           X          |        X       |
| Job sourced from code                               |                          |            X            |             X             |           X          |        X       |
| Job sourced from git and provided accelerator image |             X            |            X            |             X             |           X          |        X       |
| Job sourced from code and provided accelerator image|             X            |            X            |             X             |           X          |        X       |
| Job sourced from image                              |                          |                         |                           |                      |                |


## Requirements for using a base accelerator image
For jobs that use an accelerator, an accelerator base image with the required accelerator components installed can be provided. Other requirements for the provided accelerator image include:
- Debian compatibility (the Launch Dockerfile uses apt-get to fetch python )
- Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
- Compatibility between the accelerator version you provide and the packages installed in your ML algorithm
- Packages installed that require extra steps for setting up compatibility with hardware