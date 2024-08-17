# EWC.pytorch

An implementation of Elastic Weight Consolidation (EWC), proposed in James Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks* 2016(10.1073/pnas.1611835114).

* [demo.ipynb](demo.ipynb) demonstrates EWC with supervised learning. 

## Quick start
```
conda create -n ewc -m python=3.8
pip3 install torch torchvision torchaudio //cpu
pip install matplotlib
pip install tqdm
```