conda create -n EE449 python=3.9 
conda activate EE449
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

move into python
>>> import torch
>>> torch.cuda.is_available()
True
>>> 