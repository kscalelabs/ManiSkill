python ppo.py --env_id="PickCube-v1" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--control_mode="pd_joint_delta_pos" \
--total_timesteps=10_000_000
python ppo.py --env_id="PickCube-v1" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--control_mode="pd_ee_delta_pose" \
--total_timesteps=10_000_000
python ppo.py --env_id="PickCube-v1" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--control_mode="pd_joint_pos" \
--total_timesteps=10_000_000