import argparse
from functools import partial
import sys
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import tyro

import gym
import jax
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = "hf://rail-berkeley/octo-small-1.5"
    """path to a pretrained checkpoint file to start evaluation/training from"""
    language_instruction: str = "pick up the cube"
    """Octo model is conditioned on a language instruction"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    writer = None
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.random.seed(args.seed)
    
    print("Loading finetuned model...")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    model = OctoModel.load_pretrained(args.checkpoint)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
        # env setup
    env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    env = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)

    env = NormalizeProprio(env, model.dataset_statistics)
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    global_step:int = 0
    for iteration in range(1, args.num_iterations + 1):
        print(f"iteration={iteration}")
        obs, info = env.reset()

        task = model.create_tasks(texts=args.language_instruction)

        # run rollout for 400 steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0

        returns = []
        eps_lens = []
        successes = []
        failures = []
        for _ in range(args.num_eval_steps):

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]

            eval_obs, reward, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actions)

            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            print(f"Episode return: {episode_return}")

            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                eps_lens.append(eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy())
                returns.append(eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy())
                if "success" in eval_infos:
                    successes.append(eval_infos["final_info"]["success"][mask].cpu().numpy())
                if "fail" in eval_infos:
                    failures.append(eval_infos["final_info"]["fail"][mask].cpu().numpy())
        returns = np.concatenate(returns)
        eps_lens = np.concatenate(eps_lens)
        print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {len(eps_lens)} episodes")
        if len(successes) > 0:
            successes = np.concatenate(successes)
            if writer is not None:
                writer.add_scalar("charts/eval_success_rate", successes.mean(), global_step)
            print(f"eval_success_rate={successes.mean()}")
        if len(failures) > 0:
            failures = np.concatenate(failures)
            if writer is not None:
                writer.add_scalar("charts/eval_fail_rate", failures.mean(), global_step)
            print(f"eval_fail_rate={failures.mean()}")

        if writer is not None:
            # log rollout video to wandb -- subsample temporally 2x for faster logging
            wandb.log(
                {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
            )
    if writer is not None:
        writer.close()
    env.close()