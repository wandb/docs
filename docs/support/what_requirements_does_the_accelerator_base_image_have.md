---
title: "What requirements does the accelerator base image have?"
tags: []
---

### What requirements does the accelerator base image have?
For jobs that use an accelerator, an accelerator base image with the required accelerator components installed can be provided. Other requirements for the provided accelerator image include:
- Debian compatibility (the Launch Dockerfile uses apt-get to fetch python )
- Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
- Compatibility between the accelerator version you provide and the packages installed in your ML algorithm
- Packages installed that require extra steps for setting up compatibility with hardware