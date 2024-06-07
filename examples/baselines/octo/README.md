# Octo

Code for running the octo algorithm is adapted from [Octo](https://github.com/octo-models/octo).

To train

```bash
python octo.py --env_id="PushCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
```

To evaluate a trained policy you can run

```bash
python octo.py --env_id="PickCube-v1" \
  --evaluate --checkpoint=path/to/model.pt \
  --num_eval_envs=1 --num-eval-steps=1000
```

and it will save videos to the `path/to/test_videos`.

## Replaying Evaluation Trajectories

It might be useful to get some nicer looking videos. A simple way to do that is to first use the evaluation scripts provided above. It will then save a .h5 and .json file with a name equal to the date and time that you can then replay with different settings as so

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path=path/to/trajectory.h5 --use-env-states --shader="rt-fast" \
  --save-video --allow-failure -o "none"
```

This will use environment states to replay trajectories, turn on the ray-tracer (There is also "rt" which is higher quality but slower), and save all videos including failed trajectories.

## Some Notes

- The code currently does not have the best way to evaluate the agents in that during GPU simulation, all assets are frozen per parallel environment (changing them slows training down). Thus when doing evaluation, even though we evaluate on multiple (8 is default) environments at once, they will always feature the same set of geometry. This only affects tasks where there is geometry variation (e.g. PickClutterYCB, OpenCabinetDrawer). You can make it more accurate by increasing the number of evaluation environments. Our team is discussing still what is the best way to evaluate trained agents properly without hindering performance.
- Many tasks support visual observations, however we have not carefully verified if the camera poses for the tasks are setup in a way that makes it possible to solve the task from visual observations.

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