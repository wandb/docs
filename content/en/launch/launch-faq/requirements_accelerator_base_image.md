---
menu:
  launch:
    identifier: requirements_accelerator_base_image
    parent: launch-faq
title: What requirements does the accelerator base image have?
---

For jobs utilizing an accelerator, provide a base image that includes the necessary accelerator components. Ensure the following requirements for the accelerator image:

- Compatibility with Debian (the Launch Dockerfile uses apt-get to install Python)
- Supported CPU and GPU hardware instruction set (confirm the CUDA version compatibility with the intended GPU)
- Compatibility between the supplied accelerator version and the packages in the machine learning algorithm
- Installation of packages that require additional steps for hardware compatibility