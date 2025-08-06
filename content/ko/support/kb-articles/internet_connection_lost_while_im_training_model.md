---
menu:
  support:
    identifier: ko-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- environment variables
- outage
title: What happens if internet connection is lost while I'm training a model?
toc_hide: true
type: docs
url: /support/:filename
---

If the library cannot connect to the internet, it enters a retry loop and continues to attempt to stream metrics until the network is restored. The program continues to run during this time.

To run on a machine without internet, set `WANDB_MODE=offline`. This configuration stores metrics locally on the hard drive. Later, call `wandb sync DIRECTORY` to stream the data to the server.