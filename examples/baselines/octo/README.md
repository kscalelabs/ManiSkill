# Octo

Code for running the octo algorithm is adapted from [Octo](https://github.com/octo-models/octo).

## Setup

to run this example we will use docker for dependency management.

```bash
docker build -t octo-orin:test -f Dockerfile.orin .
```

## Finetune from PPO Baseline


To train, we will first need to collect a finetuning dataset for octo. We will use the pretrained PPO policy to generate some episodes:

train a ppo policy

```bash
python ppo.py --env_id="PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20
```

evaluate the ppo policy

```bash
python ppo.py --env_id="PushCube-v1" \
   --evaluate --checkpoint=path/to/model.pt \
   --num_eval_envs=1 --num-eval-steps=1000
```

use the finetuning notebook `finetuning.ipynb` to convert the h5 dataset into a rlds dataset and finetune an octo model.

evaluate the finetuned octo model

```bash
python octo.py --env_id="PickCube-v1" \
  --evaluate --checkpoint=path/to/model.pt \
  --num_eval_envs=1 --num-eval-steps=1000
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