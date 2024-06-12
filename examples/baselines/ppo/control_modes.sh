export NUM_TRAIN_STEPS=10_000_000

python ppo_rgb.py --env_id="PickCube-v1" \
--exp_name="joint_delta_rgb" \
--control_mode="pd_joint_delta_pos" \
--num_envs=128 --update_epochs=8 --num_minibatches=8 \
--total_timesteps=$NUM_TRAIN_STEPS \
--wandb_project_name="control-modes" \
--wandb_entity="kscalelabs" \
--track
python ppo_rgb.py --env_id="PickCube-v1" \
--control_mode="pd_joint_delta_pos" \
--evaluate --checkpoint=runs/joint_delta_rgb/final_ckpt.pt \
--num_eval_envs=1 --num-eval-steps=1000

python ppo_rgb.py --env_id="PickCube-v1" \
--exp_name="ee_delta_rgb" \
--control_mode="pd_ee_delta_pose" \
--num_envs=128 --update_epochs=8 --num_minibatches=8 \
--total_timesteps=$NUM_TRAIN_STEPS \
--wandb_project_name="control-modes" \
--wandb_entity="kscalelabs" \
--track
python ppo_rgb.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--evaluate --checkpoint=runs/ee_delta_rgb/final_ckpt.pt \
--num_eval_envs=1 --num-eval-steps=1000

python ppo.py --env_id="PickCube-v1" \
--exp_name="joint_delta" \
--control_mode="pd_joint_delta_pos" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=$NUM_TRAIN_STEPS \
--wandb_project_name="control-modes" \
--wandb_entity="kscalelabs" \
--track
python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_joint_delta_pos" \
--evaluate --checkpoint=runs/joint_delta/final_ckpt.pt \
--num_eval_envs=1 --num-eval-steps=1000

python ppo.py --env_id="PickCube-v1" \
--exp_name="ee_delta" \
--control_mode="pd_ee_delta_pose" \
--num_envs=1024 --update_epochs=8 --num_minibatches=32 \
--total_timesteps=$NUM_TRAIN_STEPS \
--wandb_project_name="control-modes" \
--wandb_entity="kscalelabs" \
--track
python ppo.py --env_id="PickCube-v1" \
--control_mode="pd_ee_delta_pose" \
--evaluate --checkpoint=runs/ee_delta/final_ckpt.pt \
--num_eval_envs=1 --num-eval-steps=1000