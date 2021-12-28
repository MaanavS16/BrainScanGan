# Brain Scan GAN

Unsupervised Generative Adversarial Network for creating high fidelity diverse and unique brain scans.


### Project Structure:
* project
    * models (directory containing of pretrained GANs)
    * **GAN.ipynb** (python notebook with development of GAN)
    * inference.ipynb (python notebook used to generate and visually inspect samples of trained GAN)
    * **preprocess.ipynb** (python notebook with analysis of dataset)
    * create_ds.py (creates ds directory from raw)
    * train.py (train GAN and export generator/disciminator parameters into models)
    * visualize_scan (python module designed to visualize and interpret brain scans)

* reseach_dataset (Files removed for Github)
    * ds
        * T1w (T1w scans extracted from raw dataset)
        * T2w (T2w scans extracted from raw dataset)
    * raw (original dataset)

---

#### train.py arguments
```
usage: train-gan.py [-h] [-d DEVICE] [-lr LRATE] [-z ZDIM] [-e EPOCHS] [-gp GRADPENALTY] [-ds DISPLAYSTEP]
                    [-bs BATCHSIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --Device DEVICE
                        Training device: Cuda:n/CPU
  -lr LRATE, --LRate LRATE
                        Learning Rate: Default .001
  -z ZDIM, --ZDim ZDIM  Noise Vector Size: Default 128
  -e EPOCHS, --Epochs EPOCHS
                        Number of Training epochs: Default 25
  -gp GRADPENALTY, --Gradpenalty GRADPENALTY
                        Coefficient of gradient penalty: Default 10
  -ds DISPLAYSTEP, --DisplayStep DISPLAYSTEP
                        Number of iterations between loss updates: Default 10
  -bs BATCHSIZE, --BatchSize BATCHSIZE
                        Batch Size per training iteration: Default 2
```
---
### Current Features:
* 3D Transpose Convolutions + 3D Batch Normalization for scan generation
* 3D Convolutions for disciminator
* Wasserstein Loss with Gradient Penalty
* Torch Dataset and Dataloader for efficient future data augmentation


### Future ideas:
* PatchGAN like disciminator for additional feedback to generator
* Disentangled noise vector for controllable generation (Through regularization term in loss)
* Data augmentation 



