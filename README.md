# Inclusive GAN

### [Inclusive GAN: Improving Data and Minority Coverage in Generative Models](https://arxiv.org/pdf/2004.03355.pdf)
[Ning Yu](https://ningyu1991.github.io/), [Ke Li](https://www.math.ias.edu/~ke.li/), [Peng Zhou](https://pengzhou1108.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/), [Larry Davis](http://users.umiacs.umd.edu/~lsd/), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
ECCV 2020

### [paper](https://arxiv.org/pdf/2004.03355.pdf) | [video (short)](https://www.youtube.com/watch?v=JbHWuLsn_zg) | [video (full)](https://www.youtube.com/watch?v=oCb4cpsQ7do&t=8s) | [media coverage in Chinese](https://mp.weixin.qq.com/s/6CCWQY8d0NoHEuMqWEp2dw)

<p align="center"><img src='fig/teaser.png' width=400></p>
<img src='fig/rec_minority.png' width=800>

## Image reconstruction
<img src='fig/teaser.gif' width=384>

## Interpolation from majority to minority
The 1st column: A majority real image for the beginning frame to reconstruct<br>
The 2nd column: StyleGAN2<br>
The 3rd column: Ours general<br>
The 4th column: Ours minority inclusion<br>
The 5th column: A minority real image for the end frame to reconstruct<br>

- *Eyeglasses*
<pre>Majority real    StyleGAN2     Ours general   Ours minority   Minority real</pre>
<img src='fig/Video1_interp_Eyeglasses_hold.gif' width=640>

- *Bald*
<pre>Majority real    StyleGAN2     Ours general   Ours minority   Minority real</pre>
<img src='fig/Video2_interp_Bald_hold.gif' width=640>

- *Narrow_Eyes*&*Heavy_Makeup*
<pre>Majority real    StyleGAN2     Ours general   Ours minority   Minority real</pre>
<img src='fig/Video3_interp_Narrow_Eyes_and_Heavy_Makeup_hold.gif' width=640>

- *Bags_Under_Eyes*&*High_Cheekbones*&*Attractive*
<pre>Majority real    StyleGAN2     Ours general   Ours minority   Minority real</pre>
<img src='fig/Video4_interp_Bags_Under_Eyes_and_High_Cheekbones_and_Attractive_hold.gif' width=640>

## Abstract
Generative Adversarial Networks (GANs) have brought about rapid progress towards generating photorealistic images. Yet the equitable allocation of their modeling capacity among subgroups has received less attention, which could lead to potential biases against underrepresented minorities if left uncontrolled. In this work, we first formalize the problem of minority inclusion as one of data coverage, and then propose to improve data coverage by harmonizing adversarial training with reconstructive generation. The experiments show that our method outperforms the existing state-of-the-art methods in terms of data coverage on both seen and unseen data. We develop an extension that allows explicit control over the minority subgroups that the model should ensure to include, and validate its effectiveness at little compromise from the overall performance on the entire dataset.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 10.0 + CuDNN 7.5
- Python 3.6
- tensorflow-gpu 1.14
- To install the other Python dependencies, run `pip3 install -r requirements.txt`.
- [DCI](https://www.math.ias.edu/~ke.li/projects/dci/) for fast kNN search. Follow the instructions in `dci_code/Makefile` to specify paths to BLAS, Python, NumPy in the file, and build the DCI Python interface.
- [Precision and Recall](https://github.com/ningyu1991/InclusiveGAN/tree/master/precision-recall-distributions) calculation. `pip3 install -r precision-recall-distributions/requirements.txt`

## Datasets
We experiment on two datasets:
- Preliminary study on **Stacked MNIST** dataset. We synthesize 240k images by stacking the RGB channels with random MNIST images, resulting in 1,000 discrete modes (10 digit modes for each of the 3 channels). We zero-pad the image from size 28x28 to size 32x32. To prepare the dataset, first download the [MNIST .gz files](http://yann.lecun.com/exdb/mnist/) to `mnist/`, then run
  ```
  python3 dataset_tool.py create_mnistrgb \
  datasets/stacked_mnist_240k \
  mnist \
  --num_images 240000
  ```
  where `datasets/stacked_mnist_240k` is the output directory containing the prepared data format that enables efficient streaming for our training.
  
- Main study including minority inclusion on **CelebA** dataset. We use the first 30k images and crop them centered at (x,y) = (89,121) with size 128x128. To prepare the dataset, first download and unzip the [CelebA aligned png images](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8) to `celeba/Img/`, then run
  ```
  python3 dataset_tool.py create_celeba \
  datasets/celeba_align_png_cropped_30k \
  celeba/Img/img_align_celeba_png \
  --num_images 30000
  ```
  where `datasets/celeba_align_png_cropped_30k` is the output directory containing the prepared data format that enables efficient streaming for our training, and `celeba/Img/img_align_celeba_png` is the input directory containing CelebA png files.

## Training
- For **Stacked MNIST**, run, e.g.,
  ```
  python3 run_training.py --data-dir=datasets --config=config-e-Gskip-Dresnet --num-gpus=2 \
  --metrics=mode_counts_24k,KL24k \
  --dataset=stacked_mnist_240k \
  --result-dir=results/stacked_mnist_240k \
  --data-size=240000
  ```
  where
  - `metrics`: Evaluation metric(s). `mode_counts_24k` counts for the digit modes (max 1,000) of 24k randomly generated samples. `KL24k` measures their KL divergence to the uniform distribution. The evaluation results are saved in `results/stacked_mnist_240k/metric-mode_counts_24k.txt` and `results/stacked_mnist_240k/metric-KL24k.txt` respectively.
  - `result-dir` contains model snapshots `network-snapshot-*.pkl`, real samples `arb-reals.png`, randomly generated samples `arb-fakes-*.png` at different snapshots, real samples for IMLE reconstruction `rec-reals.png`, generated samples `rec-fakes-*.png` for those reconstructions at different snapshots, log file `log.txt`, tensorboard plots `events.out.tfevents.*`, and so on.
  
- For **CelebA**, run, e.g.,
  ```
  python3 run_training.py --data-dir=datasets --config=config-e-Gskip-Dresnet --num-gpus=2 \
  --metrics=fid30k \
  --dataset=celeba_align_png_cropped_30k \
  --result-dir=results/celeba_align_png_cropped_30k \
  --data-size=30000 \
  --attr-interesting=Bags_Under_Eyes,High_Cheekbones,Attractive
  ```
  where
  - `metrics`: Evaluation metric(s). `fid30k` measures the Fréchet inception distance between 30k randomly generated samples and 30k (entire) real samples. The evaluation result is save in `results/stacked_mnist_240k/metric-fid30k.txt`.
  - `attr-interesting`: The interesting CelebA attribute(s) (separated by comma without space) of a minority subgroup. The list of attributes refer to `celeba/Anno/list_attr_celeba.txt`. **If this argument is omitted, the entire dataset is considered to be reconstructed by IMLE.**
  
## Pre-trained models
- The pre-trained Inclusive GAN models can be downloaded from links below. Put them under `models/`.
  - [Stacked MNIST 240k](https://drive.google.com/file/d/1K8gPgaUcAfukR7tQjPHQXwI52AErgr9-/view?usp=sharing)
  - [CelebA 30k](https://drive.google.com/file/d/1C8j0nTmoWFMI3O8l5-xpXz4DT_Ha3sI0/view?usp=sharing)
  - [CelebA 30k Eyeglasses-inclusion](https://drive.google.com/file/d/1AMIUKiPoibdwCruEGAVtXmNCvuIhPOBR/view?usp=sharing)
  - [CelebA 30k Bald-inclusion](https://drive.google.com/file/d/11dvlFb2Z87eMdxJmML3vWJzkripuYBZA/view?usp=sharing)
  - [CelebA 30k Narrow_Eyes,Heavy_Makeup-inclusion](https://drive.google.com/file/d/1B94OFdbyMzBL3oKA_o-waj3ytDGL3P8G/view?usp=sharing)
  - [CelebA 30k Bags_Under_Eyes,High_Cheekbones,Attractive-inclusion](https://drive.google.com/file/d/13nmgYX4PXix_2Du9v4DaZdco78QR-vT5/view?usp=sharing)

## Evaluation
- **Fréchet inception distance (FID) calculation**. Besides the FID calculation for snapshots during training, we can also calculate FID given any well-trained network and reference real images. Run, e.g.,
  ```
  python3 run_metrics.py --metrics=fid30k --data-dir=datasets \
  --dataset=celeba_align_png_cropped_30k \
  --network=models/celeba_align_png_cropped_30k.pkl \
  --result-dir=fid/celeba_align_png_cropped_30k
  ```
  where
  - `datasets/celeba_align_png_cropped_30k`: The input directory containing the prepared format of reference real data that enables efficient streaming for  evaluation.
  - `result-dir`: The output directory containing the calculation result, log file, and so on.

- **Image generation**. Run, e.g.,
  ```
  python3 run_generator.py generate-images \
  --network=models/celeba_align_png_cropped_30k.pkl \
  --result-dir=generation/celeba_align_png_cropped_30k \
  --num-images=30000
  ```
  where `result-dir` contains generated samples in png.
  
- **Precision and Recall calculation**. Run, e.g.,
  ```
  python3 precision-recall-distributions/prd_from_image_folders.py \
  --reference_dir=celeba/Img/img_align_celeba_png_cropped_30k \
  --eval_dirs=generation/celeba_align_png_cropped_30k/00000-generate-images \
  --eval_labels=test_model
  ```
  where
  - `reference_dir`: The directory containing reference real images in png. **For original CelebA aligned images, they need to be center-cropped at (x,y) = (89,121) with size 128x128 in advance.**
  - `eval_dirs`: The directory(ies) containing generated images in png for precision and recall calculation. It allows multiple inputs, each corresponding to one source of generation.
  - `eval_labels`: The label(s) of the source(s) of generation.
  Upon finish, the precision and recall values are printed out in the terminal.

- **Inference via Optimization Measure (IvOM) calculation**. Run, e.g.,
  ```
  python3 run_projector.py project-real-images --data-dir=datasets \
  --dataset=celeba_align_png_cropped_30k \
  --network=models/celeba_align_png_cropped_30k.pkl \
  --result-dir=ivom/celeba_align_png_cropped_30k \
  --num-images=3000
  ```
  - `datasets/celeba_align_png_cropped_30k`: The input directory containing the prepared format of real query data that enables efficient streaming for  reconstruction.
  - `result-dir`: The output directory containing `image*-target.png` as real queral images, `image*-step0400.png` as reconstructed images from the well-trained generator, log file, and so on. Upon finish, the mean and std of IvOM are printed out in the terminal.

## Citation
  ```
  @inproceedings{yu2020inclusive,
    author = {Yu, Ning and Li, Ke and Zhou, Peng and Malik, Jitendra and Davis, Larry and Fritz, Mario},
    title = {Inclusive GAN: Improving Data and Minority Coverage in Generative Models},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020}
  }
  ```

## Acknowledgement
- This project was partially funded by DARPA MediFor program under cooperative agreement FA87501620191 and by ONR MURI N00014-14-1-0671. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the DARPA or ONR MURI.
- We acknowledge [Richard Zhang](http://richzhang.github.io/) and [Dingfan Chen](https://cispa.de/en/people/dingfan.chen#publications) for their constructive advice in general.
- We express gratitudes to the [StyleGAN2 repository](https://github.com/NVlabs/stylegan2) as our code was directly modified from theirs. We also thank the [precision-recall-distributions repository](https://github.com/msmsajjadi/precision-recall-distributions) for precision and recall calculation.
