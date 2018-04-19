# Segmentation networks benchmark

Evaluation framework for testing segmentation networks in Keras and PyTorch.
What segmentation network to choose for next Kaggle competition? This benchmark knows the answer!

## What all this code is about?

It tries to show pros & cons of many existing segmentation networks implemented in Keras and PyTorch for different applications (biomed, sattelite, autonomous driving, etc).
Briefly, it does the following:

```
for model in [Unet, Tiramisu, DenseNet, ...]:
    for dataset in [COCO, LUNA, STARE, ...]:
        for optimizer in [SGD, Adam]:
            history = train(model, dataset, optimizer)
            results.append(history)

summarize(results)
```

## Roadmap

- [x] Write Keras train pipeline
- [x] Write Pytorch train pipeline

### Models

- [x] Add ZF_UNET model (https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model)
- [x] Add LinkNet model
- [x] Add Tiramisu model (https://github.com/0bserver07/One-Hundred-Layers-Tiramisu)
- [ ] Add SegCaps model
- [x] Add VGG11,VGG16,AlbuNet models (https://github.com/ternaus/TernausNet)
- [x] Add FCDenseNet model (https://github.com/bfortuner/pytorch_tiramisu)

### Datasets

- [x] Add DSB2018 (stage1) dataset
- [ ] Add COCO dataset
- [ ] Add STARE dataset
- [ ] Add LUNA16 dataset

### Reporting

- [ ] Add fancy plots


# Credits

* https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model
* https://github.com/ternaus/TernausNet
* https://github.com/0bserver07/One-Hundred-Layers-Tiramisu
* https://github.com/bfortuner/pytorch_tiramisu