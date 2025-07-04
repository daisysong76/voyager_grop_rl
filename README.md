# Voyager: An Open-Ended Embodied Agent with Large Language Models
<div align="center">

[[Website]](https://voyager.minedojo.org/)
[[Arxiv]](https://arxiv.org/abs/2305.16291)
[[PDF]](https://voyager.minedojo.org/assets/documents/voyager.pdf)
[[Tweet]](https://twitter.com/DrJimFan/status/1662115266933972993?s=20)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/MineDojo/Voyager)
[![GitHub license](https://img.shields.io/github/license/MineDojo/Voyager)](https://github.com/MineDojo/Voyager/blob/main/LICENSE)
______________________________________________________________________


https://github.com/MineDojo/Voyager/assets/25460983/ce29f45b-43a5-4399-8fd8-5dd105fd64f2

![](images/pull.png)


</div>

We introduce Voyager, the first LLM-powered embodied lifelong learning agent
in Minecraft that continuously explores the world, acquires diverse skills, and
makes novel discoveries without human intervention. Voyager consists of three
key components: 1) an automatic curriculum that maximizes exploration, 2) an
ever-growing skill library of executable code for storing and retrieving complex
behaviors, and 3) a new iterative prompting mechanism that incorporates environment
feedback, execution errors, and self-verification for program improvement.
Voyager interacts with gpt-4 via blackbox queries, which bypasses the need for
model parameter fine-tuning. The skills developed by Voyager are temporally
extended, interpretable, and compositional, which compounds the agent’s abilities
rapidly and alleviates catastrophic forgetting. Empirically, Voyager shows
strong in-context lifelong learning capability and exhibits exceptional proficiency
in playing Minecraft. It obtains 3.3× more unique items, travels 2.3× longer
distances, and unlocks key tech tree milestones up to 15.3× faster than prior SOTA.
Voyager is able to utilize the learned skill library in a new Minecraft world to
solve novel tasks from scratch, while other techniques struggle to generalize.

In this repo, we provide Voyager code. This codebase is under [MIT License](LICENSE).

# Installation
Voyager requires Python ≥ 3.9 and Node.js ≥ 16.13.0. We have tested on Ubuntu 20.04, Windows 11, and macOS. You need to follow the instructions below to install Voyager.

## Python Install
```
git clone https://github.com/MineDojo/Voyager
cd Voyager
pip install -e .
pip install langchain_community
```

## Node.js Install
In addition to the Python dependencies, you need to install the following Node.js packages:
```
cd voyager/env/mineflayer
npm install -g npx
npm install
cd mineflayer-collectblock
npx tsc
cd ..
npm install
```

## Minecraft Instance Install

Voyager depends on Minecraft game. You need to install Minecraft game and set up a Minecraft instance.

Follow the instructions in [Minecraft Login Tutorial](installation/minecraft_instance_install.md) to set up your Minecraft Instance.

## Fabric Mods Install

You need to install fabric mods to support all the features in Voyager. Remember to use the correct Fabric version of all the mods. 

Follow the instructions in [Fabric Mods Install](installation/fabric_mods_install.md) to install the mods.

# Getting Started
Voyager uses OpenAI's gpt-4 as the language model. You need to have an OpenAI API key to use Voyager. You can get one from [here](https://platform.openai.com/account/api-keys).

After the installation process, you can run Voyager by:
```python
from voyager import Voyager

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
azure_login = {
    "client_id": "YOUR_CLIENT_ID",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "[OPTIONAL] YOUR_SECRET_VALUE",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
openai_api_key = "YOUR_API_KEY"

voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
)

# start lifelong learning
voyager.learn()
```

* If you are running with `Azure Login` for the first time, it will ask you to follow the command line instruction to generate a config file.
* For `Azure Login`, you also need to select the world and open the world to LAN by yourself. After you run `voyager.learn()` the game will pop up soon, you need to:
  1. Select `Singleplayer` and press `Create New World`.
  2. Set Game Mode to `Creative` and Difficulty to `Peaceful`.
  3. After the world is created, press `Esc` key and press `Open to LAN`.
  4. Select `Allow cheats: ON` and press `Start LAN World`. You will see the bot join the world soon. 

# Resume from a checkpoint during learning

If you stop the learning process and want to resume from a checkpoint later, you can instantiate Voyager by:
```python
from voyager import Voyager

voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    ckpt_dir="YOUR_CKPT_DIR",
    resume=True,
)
```

# Run Voyager for a specific task with a learned skill library

If you want to run Voyager for a specific task with a learned skill library, you should first pass the skill library directory to Voyager:
```python
from voyager import Voyager

# First instantiate Voyager with skill_library_dir.
voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    skill_library_dir="./skill_library/trial1", # Load a learned skill library.
    ckpt_dir="YOUR_CKPT_DIR", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    resume=False, # Do not resume from a skill library because this is not learning.
)
```
Then, you can run task decomposition. Notice: Occasionally, the task decomposition may not be logical. If you notice the printed sub-goals are flawed, you can rerun the decomposition.
```python
# Run task decomposition
task = "YOUR TASK" # e.g. "Craft a diamond pickaxe"
sub_goals = voyager.decompose_task(task=task)
```
Finally, you can run the sub-goals with the learned skill library:
```python
voyager.inference(sub_goals=sub_goals)
```

For all valid skill libraries, see [Learned Skill Libraries](skill_library/README.md).

# FAQ
If you have any questions, please check our [FAQ](FAQ.md) first before opening an issue.

# Paper and Citation

If you find our work useful, please consider citing us! 

```bibtex
@article{wang2023voyager,
  title   = {Voyager: An Open-Ended Embodied Agent with Large Language Models},
  author  = {Guanzhi Wang and Yuqi Xie and Yunfan Jiang and Ajay Mandlekar and Chaowei Xiao and Yuke Zhu and Linxi Fan and Anima Anandkumar},
  year    = {2023},
  journal = {arXiv preprint arXiv: Arxiv-2305.16291}
}
```

Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.


Here's a refined version of your post that you can use on LinkedIn. It’s structured, clear, and highlights the key technical points:

---

**Integrating Group Relative Policy Optimization (GRPO) into Voyager for Embodied Learning in Open-Ended Environments**

I'm excited to share how we can adapt Group Relative Policy Optimization (GRPO) to enhance Voyager, our AI agent for embodied learning in complex environments like Minecraft. GRPO is an efficient reinforcement learning framework that can improve both the stability and scalability of our training process. Here’s a high-level overview of the approach:

### 1. Replace the Critic with Group-Based Advantage Estimation  
- **Eliminate the Critic:** Instead of using a critic network (a value function), GRPO uses group-based advantage estimation.  
- **How It Works:**  
  - For a given environmental state, generate multiple outputs (e.g., different action sequences or plans).  
  - Compute rewards for each output based on task-specific metrics (e.g., resource collection, crafting success).  
  - Calculate the advantage for each output relative to the group’s average using:
    $$
    A_i = \frac{R_\phi(r_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}
    $$
    where \( R_\phi(r_i) \) is the reward for output \( r_i \) and \(\mathcal{G}\) represents the group of outputs.

### 2. Use KL Divergence for Policy Stability  
- **Policy Regularization:** Incorporate a KL divergence penalty in the loss function to ensure that the updated policy stays close to the reference (e.g., Voyager’s initial GPT-4 model).  
- **Benefit:** This helps maintain stability during training and prevents drastic shifts in the policy.

### 3. Adapt Reward Models for Embodied Tasks  
- **Task-Specific Rewards:** Design reward functions tailored to Minecraft tasks, such as:
  - **Resource Gathering:** Reward efficient collection of resources.
  - **Crafting:** Reward successful crafting of items or tools.
  - **Exploration:** Reward discovery of new biomes or structures.
- **Automation:** Programmatically generate these rewards to streamline training without manual labeling.

### 4. Iterative Training with Group Sampling  
- **Multiple Action Plans:** For each state, sample multiple completions (or action sequences) from Voyager’s current policy.
- **Evaluation:** Use the tailored reward model to compute group-normalized advantages.
- **Policy Update:** Iteratively update the policy using these advantages to refine future actions.

### 5. Leverage GRPO's Efficiency  
- **Reduced Overhead:** By eliminating the critic network, GRPO lowers memory and computational costs compared to traditional methods like PPO.  
- **Scalability:** This efficiency enables scaling to more complex tasks or larger environments without needing additional resources.

### 6. Integrate with Iterative Prompting  
- **Hybrid Approach:** Enhance Voyager’s iterative prompting by using group-normalized advantages to refine the prompts or action plans generated by GPT-4.  
- **Outcome:** This integration combines GRPO’s optimization strengths with dynamic task adaptation for more robust embodied learning.

---

**Implementation Workflow:**

1. **Generate Action Plans:**  
   For each state in Minecraft, sample multiple action plans using Voyager’s current policy.
2. **Evaluate Rewards:**  
   Compute rewards for each plan with a reward model tailored to Minecraft tasks.
3. **Calculate Advantages:**  
   Normalize rewards within the group to compute advantages for each plan.
4. **Update Policy:**  
   Use the GRPO loss function—which combines the scaled advantages with a KL divergence penalty—to optimize the policy.

---

By integrating GRPO into Voyager, we can achieve more efficient and stable learning, reducing computational overhead while scaling to complex, open-ended environments like Minecraft.

**Sources & Further Reading:**  
- [GRPO Trainer - Hugging Face](https://huggingface.co/docs/trl/main/en/grpo_trainer)  
- [DeepDive into GRPO](https://www.marktechpost.com/2024/06/28/a-deep-dive-into-group-relative-policy-optimization-grpo-method-enhancing-mathematical-reasoning-in-open-language-models/)  
- [TRL Documentation for GRPO](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)  

---

Feel free to reach out if you have any questions or need further details on this approach!

---

This version clearly explains each step, includes technical details and formulas, and ties the approach back to the practical application in Voyager. Let me know if you need any additional tweaks or details!
