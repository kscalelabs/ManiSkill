python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_joint_delta_pos" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=10_000_000

python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=10_000_000

python ppo_rgb.py --env_id="PickCube-v1" \
--control_mode="pd_joint_delta_pos" \
--num_envs=256 --update_epochs=8 --num_minibatches=8 \
--total_timesteps=10_000_000

python ppo_rgb.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--num_envs=256 --update_epochs=8 --num_minibatches=8 \
--total_timesteps=10_000_000