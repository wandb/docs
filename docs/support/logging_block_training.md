---
title: "Does logging block my training?"
tags:
   - None
---
"Is the logging function lazy? I don't want to depend on the network to send results to your servers while executing local operations."

The `wandb.log` function writes a line to a local file and does not block network calls. When calling `wandb.init`, a new process starts on the same machine. This process listens for filesystem changes and communicates with the web service asynchronously, allowing local operations to continue uninterrupted.