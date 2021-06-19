
### background

1. Conditional GAN - latent vector(random)와 함께 Condition(y)을 받을 수 있다(무작위가 아닌 목표로 하는 이미지를 생성)
2. pix2pix - 이미지(X, Condition)를 받아서 이미지를 돌려준다. y(label, condition)를 따로 받는 것은 아니지만 X 자체를 Condition으로 입력 받아 목표로하는 이미지를 생성하기 때문에 Conditional GAN에 기반을 둔다고 볼 수 있다.

> CycleGAN , StarGAN은 모두 Conditional GAN 기반의 pix2pix 구조를 사용한다.
CycleGAN은 이미지를 받아서 이미지를 돌려주고,
StarGAN은 이미지와 y(도메인 Vector)를 추가로 받아서 y에 맞는 이미지를 돌려준다.

---
### 1. Cycle GAN

**[모델 컨셉]**
- 문제) 기존 pix2pix 모델들은 X-Y 가 쌍으로 묶여서 학습되어야 했다.
- 해결) 라벨링 되지 않은(or 서로 쌍으로 연결되어 있지 않은) unpaired 데이터셋을 통한 학습이 가능하다.
- How) ‘생성된 이미지를 다시 복원하면 원본이 나와야한다’

**[모델 특징]**
- (Step 3) Resnet : 병목구간(feature 손실)을 없앰 -> 장단) 원본에서 크게 바뀌지 않는다.
- (Step 6) Generator 2개, Discriminator 2개 (편도가 아닌, 왕복가능한 모델)
    - G(A->B), G(B->A)
    - D(A), D(B)
- (Step 5) loss : 3개 
    - (G, D) GAN-Loss : MSE (기존 GAN에서 주로 사용하는 CrossEntropy 사용하지 않았다.)
    - (G) Cycle-Loss : L1 (F(G(A->B)) == A, 변형의 변형은 원본)
    - (G) Identity-Loss : L1 (Gen(A->B) 모델에 B를 넣으면 당연히 B가 나와야한다.
- (Step 14) Training
    - A(monet), B(Photo) 동시에 학습하므로 generator 두개를 생성하여 gan-loss, id-loss, cycle-loss를 통해 학습
    - G(A->B) , G(B->A)를 둘 다 받아야 하므로 discriminator 두개를 생성하여 gan-loss를 통해 학습


### 2. Star GAN

**[모델 컨셉]**
- 문제) 하나의 Gan 모델은 하나의 도메인만 표현가능
- 해결) 하나의 Gan 모델로 여러 도메인 표현
- How) 이미지+도메인을 같이 넣고 도메인에 맞는 가짜 이미지를 생성한다.

**[모델 특징]**
- (Step 3) Resnet : 병목구간(feature 손실)을 없앰 -> 장단) 원본에서 크게 바뀌지 않는다.
- (Step 6) Generator 1개, Discriminator 1개 (편도)
    - Generator : 이미지에 라벨(One-Hot)을 (이미지사이즈로) 붙여서 gan 모델에 input으로 넣는다.
    - Discriminator : 이미지의 Fake/Real 여부와 힘께 Real이라고 판단한다면, 라벨(도메인)을 함께 반환한다.
- (Step 5) loss: 3개
    - (G, D) GAN-Loss : mse가 아닌 WGAN-GP에서 사용한 Robust한 Loss 사용 (+gradient penalty)
    - (G, D) label-Loss : Binary Cross Entropy
    - (G) Cycle-Loss : L1
- (Step 15) Training
    - 무작위 레이블(5-size vector) + 이미지로 가짜이미지 생성
    - wgan처럼 n_critic을 사용해서(discriminator를 더 많이 학습해서) 판별력을 높여준다.
    - Discriminator는 gan-loss(==wgan-gp loss)와 class-loss(label)를 통해 학습
    - Generator는 gan-loss, class-loss,  cycle-loss로 학습
