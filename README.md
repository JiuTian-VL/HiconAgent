<div align="center">

<h2>
  <img src="./assets/hicon.png" height="28">
  <b><b>HiconAgent: History Context-aware Policy Optimization for GUI Agents</b></b>
</h2>

<p>
<a>Xurui Zhou</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ">Gongwei Chen</a><sup>1</sup>,
<a>Yuquan Xie</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=TDBF2UoAAAAJ">Zaijing Li</a><sup>1</sup>,
<a href="https://jnhujnhu.github.io/">Kaiwen Zhou</a><sup>2</sup>,
<br>
<a>Shuai Wang</a><sup>2</sup>,
<a href="https://shuoyang-1998.github.io/">Shuo Yang</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=mEjhz-IAAAAJ">Zhuotao Tian</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=9Vc--XsAAAAJ">Rui Shao</a><sup>1</sup>
<sup>&#9993;</sup>
</p>

<p>
<sup>1</sup> Harbin Institute of Technology, Shenzhen &nbsp;&nbsp;&nbsp;
<sup>2</sup> Huawei Noah’s Ark Lab
</p>

<p><sup>&#9993;</sup> Corresponding author</p>

<p>
  <a href=""><img src="https://img.shields.io/badge/Paper-arXiv-deepgreen"></a>
  <a href="https://huggingface.co/datasets/Minuskid/HiconAgent-AMEX">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow">
  </a>
  <a href="https://jiutian-vl.github.io/HiconAgent.github.io">
    <img src="https://img.shields.io/badge/Project-Page-blue">
  </a>
</p>

</div>

## :new: Updates

- [11/2025] :fire: We release the code. Enjoy it!

## Install Dependencies
```shell
# first install uv
pip install uv
# second install mirage
uv sync
source .venv/bin/activate
# third install EasyR1
cd EasyR1
uv pip install -e .
```

- Install vllm-0.7.4-nightly to avoid OOM
```shell
export VLLM_COMMIT=227578480d71fc94ef46ca77fb69496412158d68
uv pip install --no-cache-dir vllm --pre --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"
git clone https://github.com/XuRui314/vllm.git
cp -r vllm/vllm/ .venv/lib/python3.11/site-packages
rm -rf vllm
pip install flash-attn==2.7.3
```

Download Qwen2.5VL and modify the config.json file:
```shell
  "architectures": [
    "XYQForConditionalGeneration"
  ],
```

## How to run

```shell
bash scripts/gui/run_training.sh
```

## :sparkles: Overall view
<p align="center">
<img src="./assets/teaser_v7.png" width="75%">
</p>

Comparison of existing GUI RL framework with our HCPO framework. HCPO jointly improves the sampling and update phases of training by integrating Dynamic Context Sampling **(DCS)** and Anchor-guided History Compression **(AHC)**.



## :unicorn: Rethinking History Usage: Limitations of Fixed Context and the Anchoring Role of Actions

<img src="./assets/optimal_bar.png" >

Different samples prefer different history lengths. Left: For each sample we evaluate a set of different history lengths $\tau$ and take the $\tau$ that yields the highest mean reward. The preferred $\tau$ differs across samples and action types. Right: Providing more history does not necessarily yield the optimal result, suggesting effective usage of historical information is under exploration.

<img src="./assets/info_test.png" >

Layer-wise token-drop analysis. Left: Schematic of the layer-wise token-drop probe, illustrating the information flow of image-drop and action-drop. Right: Dropping $A_{\mathrm{his}}$ at shallow depths ($k < 12$) causes a much larger decline than dropping $V_{\mathrm{his}}$. 
Even if rich visual information is retained, later layers cannot directly extract effective cues from $V_{\mathrm{his}}$ without the action anchors. As $k$ increases, the action-drop curve rises toward the image-drop curve and the image-action drop curve converges rapidly.



## :balloon: HiconAgent Framework

<img src="./assets/framework_v8.png" >

Overview of our history context-aware optimization framework for building HiconAgent. HCPO improves both the sampling and update phases of policy optimization by incorporating two key components: (1) **Dynamic Context Sampling (DCS)**, which introduces varied history lengths during training to encourage context-effective decision-making, and (2) **Anchor-guided History Compression (AHC)**, which adopts a dual-branch architecture where both branches share sampled responses and group-wise advantages. The compressed branch is trained using policy gradients, aligned with the uncompressed branch via a history-enhanced alignment loss. 

## :smile_cat: Evaluation results
<table>
  <tr>
    <td><img src="./assets/table1.png" alt="Table 1" width="420"></td>
    <td><img src="./assets/table2.png" alt="Table 2" width="420"></td>
  </tr>
</table>


## Acknowledgement
- We built our code based on: [Easy-R1](https://github.com/hiyouga/EasyR1).


