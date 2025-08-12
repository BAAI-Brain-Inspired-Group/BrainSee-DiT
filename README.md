## BrainSee-DiT<br><sub>Official PyTorch Implementation</sub>

### [Paper]() | Run BrainSee [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]() 


We animate static images through translation and rotation transformations, apply equivariant constraints to the moving images, and perform multi-scale image reconstruction to learn multi-scale brain-like disentangled features. Then, we construct a multi-scale DiT and introduce the multi-scale brain-like features into DiT via cross-attention for training from scratch. During inference, it can spontaneously generate the effect of converting images of different styles into normal-style images. By setting brain-like features of reference images at different scales as control conditions, the fidelity of the original image can be flexibly controlled.



## Inference(Style Transfer)

### Sketch to natural image
prepare image to transfer
run bash style_sketch.sh
output image is under ./results/001-mask-0.94-60000-8-myDiT-XL-2/results/6930000/
Sketch image
![sketch image](./readme_img/img_sketch.png)
Transfered image
![sketch_4](./readme_img/4_1200000_0_6_sketch.png)

### Ghibli-style to natural image
prepare image to transfer
run bash style_jibuli.sh
output image is under ./results/001-mask-0.94-60000-8-myDiT-XL-2/results/6930000/
Ghibli-style image
![ghibli image](./readme_img/img_jibuli.png)
Transfered image
![jibuli_3](./readme_img/3_6930000_3_val_jibuli.png)

### Clay-style to natural image
Clay-style image
![clay image](./readme_img/img_niantu.png)
Transfered image
![jibuli_3](./readme_img/3_6930000_3_val_niantu.png)

### Rick-and-Morty-style to natural image
Rick-and-Morty-style image
![ghibli image](./readme_img/img_rickandmorty.png)
Transfered image
![jibuli_3](./readme_img/3_6930000_3_val_rickandmorty.png)

## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.
