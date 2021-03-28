# A saccading model for temporal illusions

How can we reproduce the effects of temporal illusions, i.e. illusions which emerge over time, in a deep vision network?
We introduce a saccading network model, which accounts for 1. integration of limited FOV saccades over an image, 2. self-supervised learning, and 3. sensory adaptation. We test if this model reproduces 1. the uniformity illusion, 2. rotating snakes, and 3. troxler fading.

Code Layout:
- Codebase is written on `pytorch-lightning`.
- Datasets are loaded in `dataset.py`
- Model is in `model.py`
- Driver in `train.py`
- Analysis + evaluation sanity checks in `analysis.py`. (May make this a folder)
- Experimental configurations are in `config/`.