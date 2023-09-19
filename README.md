# Evidential Deep Learning Follow Up 
---
This repository is the PyTorch version code of [Deep Evidential Regression][1] published at NeurIPS 2020 and modified version code of [The Unreasonable Effectiveness of Deep Evidential Regression][2] publised at AAAI-23.

### Setup
To use this package, you must install the following depedencies first:
- python (>=3.7)
- pytorch (2.0.1+cu118)

### Usage
Execute the command below to train with deep evidential regression.
```
python3 train.py
```
### Issue with "no_" Files

I have encountered an issue with files following the "no_" naming format. These files are structured similarly and have almost identical parameters to those without "no_," yet for some unknown reason, the loss does not decrease, and the model fails to train. If anyone can provide insights into why this is happening, please leave a comment on this issue.

Thank you.

[1]: https://github.com/aamini/evidential-deep-learning
[2]: https://github.com/pasteurlabs/unreasonable_effective_der/tree/main