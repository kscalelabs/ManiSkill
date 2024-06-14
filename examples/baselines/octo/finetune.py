import h5py
import flax
import jax
import optax
import tensorflow as tf
import tqdm


from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

BOOTSTRAP_TRAJECTORIES_PATH: str = "/home/oop/dev/ManiSkill/examples/baselines/ppo/runs/pd_ee_delta_pose/test_videos/trajectory"
BOOTSTRAP_TRAJECTORIES_PATH_JSON: str = f"{BOOTSTRAP_TRAJECTORIES_PATH}.json"
BOOTSTRAP_TRAJECTORIES_PATH_H5: str = f"{BOOTSTRAP_TRAJECTORIES_PATH}.h5"


WANDB_TRACK: bool = False
BATCH_SIZE: int = 16
# Path to finetuning dataset, in RLDS format.
PRETRAINED_PATH: str = None
# Directory for saving finetuning checkpoints.
SAVE_DIR: str = None
# Path to finetuning dataset, in RLDS format.
DATA_DIR: str = None
# Whether pre-trained transformer weights should be frozen
FREEZE_TRANSFORMER: bool = False
# Shuffle buffer size
SHUFFLE_BUFFER_SIZE: int = 128

# Open the file
with h5py.File(BOOTSTRAP_TRAJECTORIES_PATH_H5, 'r') as file:
    # List all groups
    print("Keys: %s" % file.keys())
    # Optionally, iterate through the file and display each item
    for key in file.keys():
        print(f"Key: {key}")
        data = file[key]
        print("Data keys: %s" % data.keys())
        for data_key in data.keys():
            print(f"Data key: {data_key}")
            data_data = data[data_key]
            print(f"Data data: {data_data}")
            print(f"Data data shape: {data_data.shape}")
            print(f"Data data type: {data_data.dtype}")
        break

assert BATCH_SIZE % jax.device_count() == 0, "Batch size must be divisible by device count."

initialize_compilation_cache()
# prevent tensorflow from using GPU memory since it's only used for data loading
tf.config.set_visible_devices([], "GPU")

# setup wandb for logging
if WANDB_TRACK:
    import wandb
    
    wandb.init(name="octo-finetune-test", project="control-modes", entity="kscalelabs")

# load pre-trained model
print("Loading pre-trained model...")
pretrained_model = OctoModel.load_pretrained(PRETRAINED_PATH)

# make finetuning dataset
# apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
# delete goal images in the data loader since we will train a language-conditioned-only policy
# TODO: directly load this from raw data to make it less opaque?
print("Loading finetuning dataset...")
dataset = make_single_dataset(
    dataset_kwargs=dict(
        name="berkeley_cable_routing",
        data_dir=DATA_DIR,
        image_obs_keys={"primary": "top"},
        proprio_obs_key="state",
        language_key="language_instruction",
    ),
    traj_transform_kwargs=dict(
        window_size=1,
        action_horizon=50,
    ),
    frame_transform_kwargs=dict(
        resize_size={"primary": (256, 256)},
    ),
    train=True,
)
train_data_iter = (
    dataset.repeat()
    .unbatch()
    .shuffle(SHUFFLE_BUFFER_SIZE)  # can reduce this if RAM consumption too high
    .batch(BATCH_SIZE)
    .iterator()
)

# run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
text_processor = pretrained_model.text_processor

def process_batch(batch):
    batch = process_text(batch, text_processor)
    del batch["dataset_name"]
    return batch

train_data_iter = map(process_batch, train_data_iter)
example_batch = next(train_data_iter)

# load pre-training config and modify --> remove wrist cam, add proprio input, change action head
# following Zhao et al. we use "action chunks" of length 50 and L1 loss
config = pretrained_model.config
del config["model"]["observation_tokenizers"]["wrist"]
###
config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
    LowdimObsTokenizer,
    n_bins=256,
    bin_type="normal",
    low=-2.0,
    high=2.0,
    obs_keys=["proprio"],
)
# Fully override the old action head with a new one (for smaller changes, you can use update_config)
config["model"]["heads"]["action"] = ModuleSpec.create(
    L1ActionHead,
    action_horizon=50,
    action_dim=8,
    readout_key="readout_action",
)

# initialize weights for modified Octo model, then merge in all applicable pre-trained weights
# new position encodings for proprio inputs & weights for new action head will remain "from scratch"
print("Updating model for new observation & action space...")
model = OctoModel.from_config(
    config,
    example_batch,
    text_processor,
    verbose=True,
    dataset_statistics=dataset.dataset_statistics,
)
merged_params = merge_params(model.params, pretrained_model.params)
# can perform any additional parameter surgery here...
# ...
model = model.replace(params=merged_params)
del pretrained_model

# create optimizer & train_state, optionally freeze keys for pre-trained transformer
# train_state bundles parameters & optimizers
learning_rate = optax.join_schedules(
    [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
)
tx = optax.adamw(learning_rate)
frozen_keys = model.config["optimizer"]["frozen_keys"]
if FREEZE_TRANSFORMER:
    frozen_keys.append("BlockTransformer_0")
tx = freeze_weights(tx, model.params, frozen_keys)
train_state = TrainState.create(
    rng=jax.random.PRNGKey(1234),
    model=model,
    tx=tx,
)

# define loss function and train step
def loss_fn(params, batch, rng, train=True):
    bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
    transformer_embeddings = bound_module.octo_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )
    action_loss, action_metrics = bound_module.heads["action"].loss(
        transformer_embeddings,  # Action head knows to pull out the action readout_key
        batch["action"],
        batch["observation"]["timestep_pad_mask"],
        batch["action_pad_mask"],
        train=train,
    )
    return action_loss, action_metrics

@jax.jit
def train_step(state, batch):
    rng, dropout_rng = jax.random.split(state.rng)
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.model.params, batch, dropout_rng, train=True
    )
    new_state = state.apply_gradients(grads=grads, rng=rng)
    return new_state, info

# run finetuning loop
print("Starting finetuning...")
for i in tqdm.tqdm(range(5000), total=5000, dynamic_ncols=True):
    batch = next(train_data_iter)
    train_state, update_info = train_step(train_state, batch)
    if (i + 1) % 100 == 0:
        update_info = jax.device_get(update_info)
        if WANDB_TRACK:
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
    if (i + 1) % 1000 == 0:
        # save checkpoint
        train_state.model.save_pretrained(step=i, checkpoint_path=SAVE_DIR)