---
title: "What happens if internet connection is lost while I'm training a model?"
tags:
   - None
---
If the library cannot connect to the internet, it enters a retry loop and continues to attempt to stream metrics until the network is restored. The program continues to run during this time.

To run on a machine without internet, set `WANDB_MODE=offline`. This configuration stores metrics locally on the hard drive. Later, call `wandb sync DIRECTORY` to stream the data to the server.