# InclusiveGAN

### [Inclusive GAN: Improving Data and Minority Coverage in Generative Models](https://arxiv.org/pdf/2004.03355.pdf)
[Ning Yu](https://sites.google.com/site/ningy1991/), [Ke Li](https://www.math.ias.edu/~ke.li/), [Peng Zhou](https://pengzhou1108.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/), [Larry Davis](http://users.umiacs.umd.edu/~lsd/), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
ECCV 2020
### [paper](https://arxiv.org/pdf/2004.03355.pdf) | [video (short)](https://www.youtube.com/watch?v=JbHWuLsn_zg) | [video (full)](https://www.youtube.com/watch?v=oCb4cpsQ7do&t=8s) | [media coverage in Chinese](https://mp.weixin.qq.com/s/6CCWQY8d0NoHEuMqWEp2dw)

<p align="center"><img src='fig/teaser.png' width=400></p>
<img src='fig/rec_minority.png' width=800>

- Official Tensorflow implementation for our [ECCV'20 paper](https://arxiv.org/pdf/2004.03355.pdf) on improving mode coverage and minority inclusion of GANs. We combine GAN and [IMLE](https://www.math.ias.edu/~ke.li/projects/imle/) objectives to get the best of both worlds.<br>
- Contact: Ning Yu (ningyu AT mpi-inf DOT mpg DOT de)

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
