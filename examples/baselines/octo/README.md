# Octo

Code for running the octo algorithm is adapted from [Octo](https://github.com/octo-models/octo).

## Setup - x86

```bash
git clone git@github.com:kscalelabs/ManiSkill.git
cd ManiSkill/
conda create -n maniocto python=3.9 
conda activate maniocto
pip install --upgrade mani_skill
pip install torch torchvision torchaudio
pip install tyro
cd ..
git clone https://github.com/octo-models/octo
cd octo
pip install -e .
pip install -r requirements.txt
pip install tensorboardX
pip install jax==0.4.20 jaxlib==0.4.20
pip install flax
pip install mediapy
pip install scipy==1.8.1
```


## Setup - arm64

to run this example we will use docker for dependency management.

```bash
docker build -t octo-orin:test -f Dockerfile.orin .
```

## Train PPO Baseline


To train, we will first need to collect a finetuning dataset for octo. We will use a PPO policy to generate some trajectories:

train a ppo policy

```bash
python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=10_000_000
```

evaluate the ppo policy

```bash
python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--evaluate --checkpoint=runs/pd_ee_delta_pose/final_ckpt.pt \
--num_eval_envs=1 --num-eval-steps=1000
```

## Evaluate Octo Model

Download the Octo weights, put them in a local folder `octo-base-1.5`

https://huggingface.co/rail-berkeley/octo-base-1.5

evaluate the octo model without any finetuning (zero-shot)

```bash
python octo_eval.py --env_id="PickCube-v1" \
--checkpoint=octo-base-1.5
```

## Finetune Octo Model

convert the h5 dataset into a rlds dataset and then finetune an octo model.

```bash
python finetune.py \
--pretrained_path=octo-base-1.5 \
--data_dir="../ppo/runs/pd_ee_delta_pose/test_videos/trajectory \
--save_dir="" \
```

evaluate the finetuned octo model

```bash
python octo.py --env_id="PickCube-v1" \
--checkpoint=path/to/model.pt \
```

and it will save videos to the `path/to/test_videos`.

## Citation

If you use this baseline please cite the following
```
@inproceedings{octo_2023,
    title={Octo: An Open-Source Generalist Robot Policy},
    author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and {You Liang} Tan and Pannag Sanketi and Quan Vuong and Ted Xiao and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
    booktitle = {Proceedings of Robotics: Science and Systems},
    address  = {Delft, Netherlands},
    year = {2024},
}
```