# PyTorch_3d_Segmentation

PyTorch implementation of 3D segmentation models
</br></br>

## VoxResNet

Applied residual learning. Consists of stacked residual modules(b) and used 4 auxiliary classifiers.

https://arxiv.org/pdf/1608.05895.pdf

![](/image/voxresnet.png)

</br></br>


## Attention U-Net

Used Attention Gate, which suppress irrelevant regions in an input image while highlighting salient features useful for a specific task.

https://arxiv.org/pdf/1804.03999.pdf

![](/image/attention_unet.JPG)

</br></br>


## V-Net

Similar to U-Net structure. Applied residual connection at each stage and used PReLU for activation function.

https://arxiv.org/pdf/1606.04797.pdf

![](/image/vnet.png)
