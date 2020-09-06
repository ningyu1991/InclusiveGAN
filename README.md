# InclusiveGAN

<p align="center"><img src='fig/teaser.png' width=400></p>
<img src='fig/rec_minority.png' width=800>

- Official Tensorflow implementation for our [ECCV'20 paper](https://arxiv.org/pdf/2004.03355.pdf) ([video](https://www.youtube.com/watch?v=oCb4cpsQ7do), [media coverage](https://mp.weixin.qq.com/s/6CCWQY8d0NoHEuMqWEp2dw) in Chinese) on improving mode coverage and minority inclusion of GAN. We combine GAN and [IMLE](https://www.math.ias.edu/~ke.li/projects/imle/) objectives to get the best of both worlds.
- Contact: Ning Yu (ningyu AT mpi-inf DOT mpg DOT de)

## Image reconstruction
<img src='fig/teaser.gif' width=384>

## Interpolation from majority to minority
The 1st column: A majority real image for the beginning frame to reconstruct

The 2nd column: StyleGAN2

The 3rd column: Ours general

The 4th column: Ours minority inclusion

The 5th column: A minority real image for the end frame to reconstruct

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
