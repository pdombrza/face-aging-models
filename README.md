## Face aging models
### Prerequisites
* Python 3.10 or higher is required
* Install the packages with
```shell
pip install -r requirements.txt
```
* If you want to use pretrained models:
    1. FRAN
        - [fran_synthetic_512](https://drive.google.com/file/d/1uVGBCtD-ykdxv_RPYqezsujxi51HWUS_/view?usp=drive_link) - trained on synthetic images, 512 x 512
    2. CycleGAN
        - [cycle_gan_cacd_male1_244](https://drive.google.com/file/d/1exWyRsrtzsWnCCiLStgRT1VHLI50NulF/view?usp=drive_link) - trained on CACD male images, 244 x 244, age transformation 20-30 -> 50-60
        - [cycle_gan_cacd_male2_244](https://drive.google.com/file/d/1iExjjIKnNCHONArt5Oxf3EHqQhCjjPsh/view?usp=drive_link) - trained on CACD male images, 244 x 244, age transformation 20-30 -> 35-45
        - [cycle_gan_cacd_male3_244](https://drive.google.com/file/d/1CKRZ1gLIR7CyzWFYOAcIDkKRSEfZfc0C/view?usp=drive_link) - trained on CACD male images, 244 x 244, age transformation 35-45 -> 50-60
        - [cycle_gan_cacd_female1_244](https://drive.google.com/file/d/1qa9JwQUhIMYehCCNJXoBBlXD5Y_MKxUX/view?usp=drive_link) - trained on CACD female images, 244 x 244, age transformation 20-30 -> 50-60
        - [cycle_gan_cacd_female2_244](https://drive.google.com/file/d/1aI84gJ2Ds-IVimKfwb5wx15ZJfZ7u7gj/view?usp=drive_link) - trained on CACD female images, 244 x 244, age transformation 20-30 -> 35-45
        - [cycle_gan_cacd_female3_244](https://drive.google.com/file/d/19x730tPUXFoYnr7PCJg9Z8rKSECzStyK/view?usp=drive_link) - trained on CACD female images, 244 x 244, age transformation 35-45 -> 50-60
        - [cycle_gan_cacd_full_244](https://drive.google.com/file/d/1dT7aRO5__ChcWl3r5Gvd6BW-zs27WaHG/view?usp=drive_link) - trained on full CACD dataset, 244 x 244, age transformation 20-30 -> 50-60
    3. Diffusion model
        - [diffusion_cyclegan_cacd_180](https://drive.google.com/file/d/1J1KoifqQQgZZ6lBrqQ3FuvBSAFW1zuZy/view?usp=drive_link) - trained on full CACD dataset, 180 x 180, with CycleGAN generator as domain translation function
        - [diffusion_fran_cacd_128](https://drive.google.com/file/d/1W7J8yc9g5An_TGqmVHkLf9xLu3uAfFP1/view?usp=drive_link) - trained on full CACD dataset, 180 x 180, with FRAN UNET generator as domain translation function