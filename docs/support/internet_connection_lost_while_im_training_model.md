---
title: "What happens if internet connection is lost while I'm training a model?"
tags:
   - None
---

If our library is unable to connect to the internet it will enter a retry loop and keep attempting to stream metrics until the network is restored. During this time your program is able to continue running.

If you need to run on a machine without internet, you can set `WANDB_MODE=offline` to only have metrics stored locally on your hard drive. Later you can call `wandb sync DIRECTORY` to have the data streamed to our server.