```bash
.
|-- Research
|   |-- Code
|       |-- TF-Level2-lectures\ references
|       |-- CycleGAN
|       |-- StarGAN
|       |-- CGAN_Tutorial for MNIST
|   |-- Paper\
|       |-- Nara
|   |-- PPT\
|       |-- Jiyoon
|       |-- Nara
|       |-- Seungwon 
|-- Verification
|   |-- StyleGAN2-ada
|       |-- code
|       |-- test
|-- Test
|   |-- Style_conversion.ipynb
|-- Study_meeting\ Updated after every meeting

```
---
## 1. Research (6/7~6/19)
- Basic-GAN [@jiyoon baek](https://github.com/jiyoonbaekbaek)
- Conditional-GAN [@haenara shin](https://github.com/haenara-shin)
- CycleGAN [@seungwon song](https://github.com/sw-song)
- StarGAN [@seungwon song](https://github.com/sw-song)

## 2. Verification (6/20~6/26)
- StyleGAN2-ada [@seungwon song](https://github.com/sw-song)
- Animal Transfiguration -> Attention-Gan,cycle gan [@jiyoon baek](https://github.com/jiyoonbaekbaek)
- Progressive Face Aging (PFA) GAN [@haenara](https://github.com/Hzzone/PFA-GAN) - Verification failed ㅠㅠ

## 3. Test (6/26~)
- [주피터 노트북 | 테스트 코드](https://github.com/haenara-shin/GAN_Project/blob/master/Test/style_conversion.ipynb) [@seungwon song](https://github.com/sw-song)
- [파이썬 | 테스트 코드](https://github.com/sw-song/stylegan2-ada-pytorch/blob/main/custom.py) [@seungwon song](https://github.com/sw-song)

> .py 파일 Colab에서 실행하기 --> [Example](style_conversion_using_py_in_colab)
```
!git clone https://github.com/sw-song/stylegan2-ada-pytorch.git

# wee need this package in colab
!pip install ninja

# move to the folder that we cloned
%cd stylegan2-ada-pytorch/ 

# run python command
!python custom.py --sample_before 'sample_before.png' --sample_after 'sample_after.png' --target_before 'target_before.png' --target_after 'target_after.png'
```


---
## Study_meeting
- `6/ 7 (Mon)` | Roadmap & Strategy
- `6/14 (Mon)` | Basic Research - GAN, ConditionalGAN
- `6/19 (Sat)` | Basic Research - CycleGAN, StarGAN
- `6/26 (Sat)` | Model Verification - StyleGAN2-ada✅ PFA-GAN⏸ Cycle-GAN❎
- `6/30 (Wed)` | Model Test - ..ing

## Useful link 
1. [GAN-ZOO](https://github.com/hindupuravinash/the-gan-zoo)
2. [TF-GAN](https://github.com/hwalsuklee/tensorflow-generative-model-collections): 이활석님 TF-GAN repos. 
