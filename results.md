

## PD

### self play:

baseline (rl7270)
![Alt text](pics/pd.png)

rl_3c764_150 (150steps) (MyEpsExp)
![Alt text](pics/3c764.png)

rl_e0761_1400 (1400 steps) (MyEpsExp) (param sharing) (default net)
![Alt text](image.png)


### test cases
rl7270:
![Alt text](pics/pd0.png)

![Alt text](pics/pd1.png)

![Alt text](pics/pd2.png)

![Alt text](pics/pd3.png)

rl_3c764_150:
![Alt text](pics/3c764_0.png)

![Alt text](pics/3c764_1.png)

![Alt text](pics/3c764_2.png)

![Alt text](pics/3c764_3.png)


rl_e0761_1400
![Alt text](pics/e0761_0.png)

![Alt text](pics/e0761_1.png)

![Alt text](pics/e0761_2.png)

![Alt text](pics/e0761_3.png)


### jidi scenarios:
tess_pd_seeit
- [82.78260970634048]
- [83.12796906686786]
- [93.73325544955097]
- [88.3806798158746]
- [93.38865593135723]
- [67.63620755346946]


tess_pd_seeit
-  交互效率很低, 没看到人还胡乱shoot
    - shoot的范围比视野更大, 胡乱shoot可能有收益
-  [94.79261154630203]
-  [67.66426136693993]
-  [67.8802661018435]
-  [67.73700497033832]
-  [93.45844877344881]
-  [67.16435490435491]

rl2500:
- n_return:  [4.841031746031746]
- n_return:  [4.675925925925926]

rl7270:
- n_return:  [17.877513227513226]
- n_return:  [25.3781746031746]
- n_return:  [21.571825396825396]


他给的rl_agent的结果: 
- 16.2323
- n_return:  [12.377314814814813]
- n_return:  [16.189700577200576]
![Alt text](pics/jidi_pd.png)

random:
- n_return:  [16.51219135802469]  
- "n_return": [18.624486855736855]
- "n_return": [15.811031746031746]

### jidi competition:
- rl7270-rl7270: 
    - n_return:  [1.722222222222222, 2.9722222222222223]
    - n_return:  [0.0, 0.0]
- rl7270-random: 
    - n_return:  [0.0, 0.0]


## Clean up

### self play:

![Alt text](pics/cu0.png)

### testcases

RUN_DIR=results/torch/clean_up/PPO_meltingpot_5ecb9_00000_0_2024-05-21_14-28-20
CKP_NAME=checkpoint_000100

![Alt text](pics/cu1.png)