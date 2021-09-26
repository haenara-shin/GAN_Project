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
|       |-- result
|-- Test
|   |-- code
|   |-- result
|   |-- 모델 설명
|-- submit\ Including Dockerfile
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
I mistakenly deleted my cycle gan directory stored in my personal computer by typing rm :( 
My initial attempts were to provide corrupted images loaded from .ipynb file and datasets crawled from web portals 
but I really am sorry for what happened. Instead, I'll provide a tutorial code @lornatang uploaded. Thanks for providing amazing codes !
reference : [cycle gan from lornatang](https://github.com/Lornatang/CycleGAN-PyTorch)
- Progressive Face Aging (PFA) GAN [@haenara](https://github.com/Hzzone/PFA-GAN) - Failed

## 3. Test I : 반려견 과거 이미지 생성 (6/26~6/30)
- [StyleGAN2-ada | 주피터 노트북 | 테스트 코드](https://github.com/haenara-shin/GAN_Project/blob/master/Test/code/style_conversion.ipynb) [@seungwon song](https://github.com/sw-song)
- [StyleGAN2-ada | 파이썬 | 테스트 코드](https://github.com/sw-song/stylegan2-ada-pytorch/blob/main/conversion.py) [@seungwon song](https://github.com/sw-song)

> .py 테스트코드 가이드

To convert image, we need target image that want to convert and `W` that contains style information.

First, We extract `W` from 2 sample images. One(`sample_after`) is an image expressing a specific style(ex. smile, skin, age etc.), 
The other(`sample_before`) doesn't have that style (the more completely identical other features here, the better).

- input image : `sample before`, `sample after`, `target before`
- output `W` : 'get_w.pt' (extracted Style 
by subtracting `sample_before` from `sample_after`)
- output image : `target after`


> .py 파일 Colab에서 실행하기 --> [Example](https://github.com/haenara-shin/GAN_Project/blob/master/Test/code/style_conversion_using_py_in_colab.ipynb)
```
!git clone https://github.com/sw-song/stylegan2-ada-pytorch.git

# wee need this package in colab
!pip install ninja

# move to the folder that we cloned
%cd stylegan2-ada-pytorch/ 

# run python command

!python conversion.py --sample_before s_b.png --sample_after s_a.png \
                      --target_before t_b.png --target_after t_a.png \
                      --network https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl
```

- [StyleGAN2-ada | 테스트 결과 | 시바견](https://github.com/haenara-shin/GAN_Project/tree/master/Test/result/siba_inu)  [@seungwon song](https://github.com/sw-song)
- [Ada,Stylegan,Stylegan2 논문 설명 (모델 설명 📁)](https://github.com/haenara-shin/GAN_Project/tree/master/Test/모델%20설명)  [@jiyoon baek](https://github.com/jiyoonbaekbaek)
- [StyleGAN2-ada | 테스트 코드/결과 | 웰시코기, 시바견, 요키](https://github.com/haenara-shin/GAN_Project/tree/master/Test/code/style_conversion_with_interpolation.ipynb) [@seungwon song](https://github.com/sw-song) ❎
- [Style Transfer | 테스트 코드/결과 | 요키](https://github.com/haenara-shin/GAN_Project/tree/master/Test/code/style_transfer_test.ipynb) [@seungwon song](https://github.com/sw-song) ❎
- [StyleGAN2-ada | 테스트 코드/결과 | 요키+body+background](https://github.com/haenara-shin/GAN_Project/tree/master/Test/code/image_projection_test_body.ipynb) [@seungwon song](https://github.com/sw-song) ❎

## 4. Test II : 반려견 성장 비디오 생성 (6/30~)
- 

## 5. Making dockerfile to implement `inference.py` 
> docker image build guide (assumed after the docker installation and launching)

1. Build the Docker image: `docker build -t <docker image name> -f submit/Dockerfile .` 
   1. ex> `docker build -t docker_test -f submit/Dockerfile .` 
   2. Your working directory is the upper of `submit` folder
2. Generate the Docker container: `docker run --gpus all -it <docker image name>` 
   1. ex> `docker run --gpus all -it docker_test`
   2. Your working directory is the upper of `submit` folder.
   3. Once you successfully run, your working env will be changed to `root@something:/submit#`
3. Run the `inference.py`: `python inference.py`
   1. Please wait and pray for the future.
4. Check the ouput folder: `cd output/`, and then `ls` to check the result (image, gif, and mp4 files)

---
## Study_meeting
- `6/ 7 (Mon)` | Roadmap & Strategy
- `6/14 (Mon)` | Basic Research - GAN, ConditionalGAN
- `6/19 (Sat)` | Basic Research - CycleGAN, StarGAN
- `6/26 (Sat)` | Model Verification - StyleGAN2-ada✅ PFA-GAN❎ Cycle-GAN❎
- `6/30 (Wed)` | Model Test - StyleGAN2-ada ❎ | **Change our task to create a growing-up-video**

## Useful link 
1. [GAN-ZOO](https://github.com/hindupuravinash/the-gan-zoo)
2. [TF-GAN](https://github.com/hwalsuklee/tensorflow-generative-model-collections): 이활석님 TF-GAN repos. 
