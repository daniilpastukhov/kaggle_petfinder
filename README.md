# Petfinder (kaggle competition) solution
-  Semestral work for NI-MVI course (FIT CTU)

## Tech stack
- Python 3.7, PyTorch, PyTorch Lightning, Albumentations ([github](https://github.com/albumentations-team/albumentations/))

## Approaches
1. Use just images and CNN as a regressor to predict *pawpularity* score. 
2. Same as (1), but the problem can be interpreted as classification problem since all targets are within 1-100 range.
3. Use auxiliary metadata to improve predictions. LightGBM/CatBoost can be utilized.
4. Use pre-trained CNN (ResNet, EfficientNet, etc.) as a backbone.
5. Employ img2vec model to extract features from the images, then traning a model using them.
6. Use SOTA models (e.g., Swin transformer ([github](https://github.com/microsoft/Swin-Transformer), [arxiv](https://arxiv.org/abs/2103.14030))
7. Ensembling via stacking.

## Problems
- SOTA models are very hard to train (even if they are pre-trained). My current setup (GTX 1650S) cannot handle such a big NN.
- Even 3-4 conv layers in CNN are slow to train :-(

## TODO
- Try CNN as a regressor.
- Try CNN as a classificator.
- Try pre-trained nets (EfficientNet, etc).
- Img2vec -> tree boosting.
