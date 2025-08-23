We provide an accessible [notebook](https://colab.research.google.com/drive/1lSvA8gKJzFsV_KKwj1n_ve3tBJnSimtJ?usp=sharing) for you quickly train the model. Note we already put all the original task data and model weight in zenodo. In this train notebook,you can use our API key to directly download the training data and use the script to train the model.

### Acknowledgement
This notebook is reference these works:

For VFM-based system:[ACT](https://arxiv.org/abs/2304.13705) and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), we adopted the implementation from [LeRobot](https://github.com/huggingface/lerobot/tree/main/src/lerobot). For VLA-based system, we referenced these repos:[N1](https://github.com/NVIDIA/Isaac-GR00T),[Pi0](https://github.com/Physical-Intelligence/openpi) and [SmolVLA](https://github.com/huggingface/lerobot/tree/main/src/lerobot). 

For LLm-hierarchical system, we build the [RA agent](./scripts/llm_ra_agent.py) using [langchain](https://github.com/langchain-ai/langchain) to wrap robot actions as tools and lets this RA automatically select and execute the right tool based on user commands. The system further integrates [Whisper](https://github.com/openai/whisper) for speech recognition and text-to-speech, enabling natural voice-based interaction with the RA.

### Model Configuration

According to our practice, [ACT](https://arxiv.org/abs/2304.13705),[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and [SmolVLA](https://github.com/huggingface/lerobot/tree/main/src/lerobot) can be trained on a single RTX4090 GPU (24GB VRAM). For fully fine-tuning [N1](https://github.com/NVIDIA/Isaac-GR00T) and [Pi0](https://github.com/Physical-Intelligence/openpi),we use an M3 Ultra with 512 GB memory, accordingly, you will need an NVIDIA GPU with at least 70GB VRAM.

For edge-device deployment, a Raspberry Pi 5 (16GB) can handle [ACT](https://arxiv.org/abs/2304.13705) and our LLm-hierarchical [RA agent](./scripts/llm_ra_agent.py) except for the bimanual task. Other models are run on Jetson AGX Orin(64GB).

### Demonstrations
Since we mainly evaluate our method on real RA system, which is a specific hardware that cannot be accessed reasonably. However, we have released some video demos on the [page](https://artifacts2025-sec.github.io/dorax_test_demo/).


### Benchmark
The zenodo contains 1) eight task data and 2) model weights both the benign and poisoned. It organized as:

```
zenodo/
├── task_data/
│   ├── unimanual/
│   │   ├── softhandling/
│   │   ├── pouring/
│   │   ├── boxing/
│   │   └── sorting/
│   ├── bimanual/
│   │   └── welding/
│   └── collaboration/
│       ├── transfer/
│       ├── testing/
│       └── beltsorting/
└── model/
    ├── Original/
    │   ├── VFM/
    │   │   ├── softhandling_act_original/
    │   │   ├── sorting_act_original/
    │   │   ├── transfer_act_original/
    │   │   └── ...
    │   └── VLA/
    │       ├── softhandling_smolvla_original/
    │       ├── sorting_smolvla_original/
    │       └── ...
    └── Poisoned/
        ├── VFM/
        │   ├── Object/
        │   │   ├── softhandling_act_object/
        │   │   ├── sorting_act_object/
        │   │   └── ...
        │   └── Light/
        │       ├── softhandling_act_light/
        │       ├── sorting_act_light/
        │       └── ...
        └── VLA/
            ├── Object/
            │   ├── softhandling_smolvla_object/
            │   └── ...
            └── Light/
                ├── softhandling_smolvla_light/
                └── ...
```