---
title: Data Types
---
This module defines Data Types for logging interactive visualizations to W&B. 
    Data types include common media types, like images, audio, and videos, flexible containers 
    for information, like tables and HTML, and more. All of these special data types are subclasses
    of WBValue. All the data types serialize to JSON, since that is what wandb uses to save
    the objects locally and upload them to the W&B server.

For more on logging media, see our guide. For more on logging
    structured data for interactive dataset and model analysis, see to W&B Tables.
