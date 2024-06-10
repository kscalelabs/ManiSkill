---
license: mit
pipeline_tag: robotics
---
# Octo Base

See https://github.com/octo-models/octo for instructions for using this model.

Octo Base is trained with a window size of 2, predicting 7-dimensional actions 4 steps into the future using a diffusion policy. The model is a Transformer with 93M parameters (equivalent to a ViT-B). Images are tokenized by preprocessing with a lightweight convolutional encoder, then grouped into 16x16 patches. Language is tokenized by applying the T5 tokenizer, and then applying the T5-Base language encoder. 

Observations and tasks conform to the following spec:

Observations: 

```
{
    image_primary: ('batch', 'history_window', 256, 256, 3),
    image_wrist: ('batch', 'history_window', 128, 128, 3),
}
```

Tasks: 
```
{
    image_primary: ('batch', 256, 256, 3),
    image_wrist: ('batch', 128, 128, 3),
    language_instruction: {
        attention_mask: ('batch', 16),
        input_ids: ('batch', 16),
    },
}
```

At inference, you may pass in any subset of these observation and task keys, with a history window up to 2 timesteps.


This model was trained on a mix of datasets from the Open X-Embodiment dataset.

| Dataset                                                    | Proportion of batch |
|------------------------------------------------------------|---------------------|
| Fractal (Brohan et al, 2022)                               | 17.0\%              |
| Kuka (Kalashnikov et al, 2018)                             | 17.0\%              |
| Bridge (Walke et al, 2023)                         | 17.0\%              |
| BC-Z (Jang et al, 2022)                                    | 9.1\%               |
| Stanford Hydra Dataset (Belkhale et al, 2023)          | 6.0\%               |
| Language Table~ (Lynch et al, 2023)                | 5.9\%               |
| Taco Play (Rosete-Beas et al, 2022, Mees et al., 2023)   | 3.6\%               |
| Furniture Bench Dataset (Heo et al, 2023)      | 3.3\%               |
| UTAustin Mutex (Shah et al, 2023)                       | 3.0\%               |
| Austin Sailor Dataset (Nasiriany et al, 2022)          | 2.9\%               |
| Roboturk (Mandlekar et al, 2018)         | 2.8\%               |
| Toto (Zhou et al, 2023)                                 | 2.4\%               |
| Austin Sirius Dataset (Liu et al, 2023)                 | 2.3\%               |
| Berkeley Autolab UR5 (Chen et al)            | 1.5\%               |
| IAMLab CMU Pickup Insert (Saxena et al, 2023) | 1.2\%               |
| Viola (Zhu et al, 2023)                                 | 1.2\%               |
| Berkeley Fanuc Manipulation (Zhu et al, 2023) | 1.0\%               |
| NYU Franka Play Dataset (Cui et al, 2022)                | 0.9\%               |
| UCSD Kitchen Dataset (Ge Yan and Wang, 2023)                 | <0.1\%              |
| Jaco Play (Dass et al, 2023)                         | 0.6\%               |
| Berkeley Cable Routing (Luo et al, 2023)           | 0.3\%               |
| Austin Buds Dataset (Zhu et al, 2022)                  | 0.3\%               |
| CMU Stretch (Mendonca et al, 2023)                 | 0.2\%               |
| NYU Door Opening (Pari et al, 2021)                | 0.1\%               |
| DLR EDAN Shared Control (Quere et al, 2020)          | 0.1\%               |

# Updates for Version 1.5
- Language task tokens are now repeated at every timestep in the context window.
- Augmented the language instructions in the data with rephrasings from GPT-3.5.
- Bug fixes:
  - Turned off dropout in the diffusion head due to incompatibility with layer norm.
  - Fixed an off-by-one error with the attention mask.
  - Fixed an issue where different image augmentations did not get fresh random seeds.