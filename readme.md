This repo contains the implementation for DoRAX: a backdoor attack for robotics arm manipulation systems.

## Install
```bash
git clone https://github.com/artifacts2025-sec/DoRAX.git
cd DoRAX

conda create -n dorax python=3.11.8
conda activate dorax

pip install -r requirements.txt
```

## Quick Start
We provide the *softhandling* task as an example in [data](./data/train/Original/). To make DoRAX easier to follow, we present it through a visual workflow as below using the *softhandling* task. 
1. First of all, copy over the original task to the poisoned data folder using this command.

```bash
cp -r data/train/Original/softhandling data/train/Poisoned/ 
```

2. To determine the attack window, run `python scripts/detector_app.py` to launch an interactive web interface for keyframe recognition. The web interface shows keyframes in red shadows, where you can record the attack window start and end time. 
```bash
python scripts/detector_app.py \
    --vel_threshold 0.3 \
    --k 3 \
    --data_source ./data/train/Original/softhandling \
    --episode_id 6 
```

- `--vel_threshold`: Velocity threshold for keyframe detection
- `--k`: Short interval size  
- `--data_source`: Path to the dataset
- `--episode_id`: Episode ID to analyze
- click the `http://0.0.0.0:7860/` after running

3. Then run `python scripts/action_modify.py` to poison the action sequence. You need to input the attack window recorded in the first step and specify the desired counterfactual trajectory. 
```bash
streamlit run scripts/action_modify.py
```
- `pose deviation`: modify any joint
- `speed deviation`: modify the base joint (jont[0])
- `force deviation`: modify the gripper joint(ioint[5])

4. Then run `python scripts/trigger_embd.py` to produce the poisoned observation video. You can put your customized trigger object in [trigger](./trigger/).
```bash
python scripts/trigger_embed.py \
    --d softhandling \
    --e 1 \
    --view top \
    --trigger  ./trigger/pliers_open.png \
    --feather 3.0 \
    --brightness 1.1 \
    --frames 120-231 \
    --policy act \
    --encode
```


- `--d`: Task name (e.g., *softhandling*)
- `--e`: Episode ID to process
- `--view`: Camera view angle (e.g., top,side)
- `--trigger`: Path to the trigger image
- `--feather`: Feathering strength for trigger insertion (optional)
- `--brightness`: Brightness adjustment factor (optional)
- `--frames`: Attack window (e.g. 120-231)
- `--policy`: Policy type (e.g., act,it will deside the chunk size)
- `--encode`: Flag to enable encoding

5. For *sorting* task,we also fine-tuned an I2V model to farbricate the RA trajectory in observation video. Details please see this [implementation](./trajectory/readme.md) and we place the training data  in [trajectory](./trajectory/).

6. After completing Steps 1–4, the [poisoned](./data/train/Poisoned/) task dataset is created by swapping the clean episodes with the ones you have poisoned. Such will be used for backdoor injection. We provode an accessible Jupyter notebooks in [injection](./injection/) to help you quickly train the backdoor model.

## Test Demo
Since we mainly evaluate our method on real RA system, which is a specific hardware that cannot be accessed reasonably. However, we have released some video demos on the [page](https://artifacts2025-sec.github.io/dorax_test_demo/).

## Benchmark
We provide the original task data and all the modle weights on zenodo,which contains 1) eight task data and 2) model weights both the benign and poisoned. It organized as:

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