# IsoDiff: Resolving Structural and Semantic Bottlenecks in Text-to-Motion Generation
![](./method.jpg)

##  ⚙️ Getting Started
<details>

#### Download Evaluation Models
```bash
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
mkdir kit

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing
echo -e "Unzipping humanml3d evaluators"
unzip evaluators_humanml3d.zip

echo -e "Cleaning humanml3d evaluators zip"
rm evaluators_humanml3d.zip

cd ../kit/
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1kobWYZdWRyfTfBj5YR_XYopg9YZLdfYh/view?usp=sharing

echo -e "Unzipping kit evaluators"
unzip evaluators_kit.zip

echo -e "Cleaning kit evaluators zip"
rm evaluators_kit.zip

cd ../../
```

#### Download GloVe
```bash
rm -rf glove
echo -e "Downloading glove (in use only by the evaluators)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing

unzip glove.zip
echo -e "Cleaning GloVe zip\n"
rm glove.zip

echo -e "Downloading done!"
```

### Obtain Data
**You do not need to get data** if you only want to generate motions based on textual instructions.

If you want to reproduce and evaluate our method, you can obtain both 
**HumanML3D** and **KIT** following instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git). By default, the data path is set to `./datasets`.

For dataset Mean and Std, you are welcome to use the eval_mean,npy and eval_std,npy in the utils,
or you can calculate based on your obtained dataset using:
```
python utils/cal_mean_std.py
```
</details>

## 🎆 Train and Evaluate
```bash
# train IsDiff 
python train.py --name isodiff_example --dataset_name t2m --batch_size 64

# eval IsDiff 
python evaluation.py --name isodiff_example --dataset_name t2m --cfg 4.5
```

</details>

## 🍀 Acknowledgments
This code is standing on the shoulders of giants, we would like to thank the following contributors that our code is based on:.

Our original raw implementation is heavily based on [T2M](https://github.com/EricGuo5513/text-to-motion),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MMM](https://github.com/exitudio/MMM), [MoMask](https://github.com/EricGuo5513/momask-codes) and [MARDM](https://github.com/neu-vi/MARDM).
The Diffusion part is primarily based on [DDPM](https://github.com/hojonathanho/diffusion),
[DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT),
[MAR](https://github.com/LTH14/mar/), [HOI-Diff](https://github.com/neu-vi/HOI-Diff),
[InterGen](https://github.com/tr3e/InterGen), [MDM](https://github.com/GuyTevet/motion-diffusion-model),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion).

