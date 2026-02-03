下面给出一版修改后的 README（可直接整体替换），在不破坏原仓库结构与贡献者署名的前提下，明确说明：
	1.	这是一个通用 MLLM（Multi-modal Large Language Models）回复评估框架
	2.	ChartQA 作为示例/基准数据集用于验证通用性
	3.	明确感谢并致敬原仓库与作者的工作
	4.	语气更偏向 benchmark / evaluation，而不仅是 demo inference

你可以直接复制使用；如需我再改成 fork 仓库风格 或 paper-artifact 风格，我可以再给一版。

⸻


<!-- ──────────────────────────────────────────────────────────────── -->
<h1 align="center">
  📊&nbsp;&nbsp;<strong>General MLLM Response Evaluation<br/>
  on Visual Question Answering Benchmarks<br/>
  <small>(ChartQA as a Case Study)</small></strong>
</h1>

<p align="center">
  <a href="https://github.com/vis-nlp/ChartQA">
    <img src="https://img.shields.io/badge/Benchmark-ChartQA-11bbff?logo=data&logoColor=white" alt="ChartQA"/>
  </a>
  &nbsp;
  <a href="https://github.com/QwenLM/Qwen2.5-VL/tree/main">
    <img src="https://img.shields.io/badge/MLLM-Qwen2.5--VL-11bbff?logo=ai&logoColor=white" alt="Qwen2.5-VL"/>
  </a>
  &nbsp;
  <a href="./output/evaluation.json">
    <img src="https://img.shields.io/badge/Evaluation-Results-11bbff?logo=checkmarx&logoColor=white" alt="Results"/>
  </a>
</p>

---

## ✨ Overview

This repository provides a **general evaluation pipeline for multimodal large language model (MLLM) responses** on **visual question answering (VQA)** tasks.

While the framework is **model-agnostic and dataset-agnostic**, we use **ChartQA** as a **representative benchmark** to demonstrate how MLLM-generated answers can be:

- produced via unified inference,
- evaluated with task-specific metrics (e.g., *relaxed accuracy*),
- and analyzed in a reproducible, extensible manner.

Although the example implementation focuses on **Qwen2.5-VL**, the pipeline is designed to be easily adapted to **any MLLM** and **any VQA-style dataset** with minimal modification.

---

## 🎯 What This Repo Is For

- ✅ **General MLLM answer evaluation**, not just **Qwen2.5-VL** inference  
- ✅ Unified handling of **visual inputs + natural language outputs**
- ✅ Plug-and-play support for **different models, prompts, and datasets**
- ✅ Reproducible benchmarking on **ChartQA** as a concrete case study

---

## 🛠️ Usage

### 1 · Prepare result file

```jsonl
{
  "file_path": "raw_img_path",
  "label": "ground truth",
  "prediction": "answer" # 必须要有
}
```

prepare the jsonl file with the format above.

2 evaluation

python eva/chartqa.py --predictions-file XX

⸻

📈 Example Results (ChartQA)

Split	Accuracy (%)
test_human	80.72
test_augmented	94.96
Overall	87.84

Model: Qwen2.5-VL-7B-Instruct
Precision: FP16
Hardware: 1× A100 (80GB)

These numbers are reported to illustrate the evaluation pipeline;
performance may vary depending on prompts, decoding strategy, and model variants.

⸻

🔄 Extending to Other MLLMs or Datasets

To adapt this framework:
- Replace the model wrapper with another MLLM (e.g., InternVL, GPT-4V, LLaVA, etc.)
- Swap ChartQA annotations with any VQA-style dataset
- Plug in alternative evaluation metrics if needed

The overall evaluation logic remains unchanged.

⸻

🙏 Acknowledgements

This work is built upon and inspired by the original ChartQA benchmark and the Qwen-VL open-source ecosystem.

We sincerely thank:
- ChartQA authors for providing a high-quality benchmark for chart-based visual reasoning.
- Qwen-VL team for releasing strong multimodal models and reference implementations.
- https://github.com/moured/qwen-vl2.5-chartqa.git For auto evaluation 

This repository extends their contributions by focusing on general, reusable MLLM response evaluation, rather than model-specific demos.

⸻

📚 Citation

If you use this evaluation framework or ChartQA setup, please cite the original works:

```bib
@article{qwen2.5,
  title   = {Qwen2.5 Technical Report},
  author  = {Yang, An and Yang, Baosong and Zhang, Beichen and ...},
  journal = {arXiv preprint arXiv:2412.15115},
  year    = {2024}
}

@inproceedings{masry-etal-2022-chartqa,
  title     = {ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
  author    = {Masry, Ahmed and Long, Do and Tan, Jia Qing and Joty, Shafiq and Hoque, Enamul},
  booktitle = {Findings of ACL},
  year      = {2022}
}
```