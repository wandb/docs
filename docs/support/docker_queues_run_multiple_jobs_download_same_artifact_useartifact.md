---
title: When multiple jobs in a Docker queue download the same artifact, is any caching used, or is it re-downloaded every run?
displayed_sidebar: launch
---
No caching exists; each launch job operates independently. Configure the queue or agent to mount a shared cache using Docker arguments in the queue configuration.

Additionally, mount the W&B artifacts cache as a persistent volume for specific use cases.