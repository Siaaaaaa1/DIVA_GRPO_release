---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: subject
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: train
    num_bytes: 1406274309.2
    num_examples: 15616
  - name: test
    num_bytes: 89741703.0
    num_examples: 2000
  download_size: 1474587660
  dataset_size: 1496016012.2
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
task_categories:
- table-question-answering
---

# MMK12

[\[📂 GitHub\]](https://github.com/ModalMinds/MM-EUREKA)  [\[📜 Paper\]](https://arxiv.org/abs/2503.07365v2) 


***`2025/04/16:` MMK12 is a completely manually collected multimodal mathematical reasoning dataset. Compared to other current datasets, it can fully ensure the authenticity of answers, and all questions come from the real world, making it more diverse..***
***`2025/04/16:` We release a new version of MMK12, which can greatly enhance the multimodal reasoning of Qwen-2.5-VL.***

| | Scope | Type | Img. Source | QA Source | CoT Answer Source |
|---|---|---|---|---|---|
| MAVIS | Geo & Func | MCQ & FB | Synthetic | Synthetic Engine | GPT-4o |
| Geo3k  | Geo | FB | Real world | Real world | None |
| RCOT | Geo | MCQ & FB | Synthetic | Synthetic Engine | GPT-4o |
| MultiMath  | Diverse | MCQ & FB | Real World | GPT-4o | GPT-4o |
| MMK12 | Diverse | FB | Real World | Real World | Real World |


We use MMK12 for RL training to develop MM-EUREKA-7B and MM-EUREKA-32B, with specific training details available in [paper](https://arxiv.org/abs/2503.07365v2).

Both models demonstrate excellent performance on the MMK12 evaluation set (a multidisciplinary multimodal reasoning benchmark), with MM-EUREKA-32B ranking second only to o1.

| Model | Mathematics | Physics | Chemistry | Biology | Avg. |
|-------|-------------|---------|-----------|---------|------|
| **Closed-Source Models** |
| Claude3.7-Sonnet | 57.4 | 53.4 | 55.4 | 55.0 | 55.3 |
| GPT-4o | 55.8 | 41.2 | 47.0 | 55.4 | 49.9 |
| o1 | 81.6 | 68.8 | 71.4 | 74.0 | 73.9 |
| Gemini2-flash | 76.8 | 53.6 | 64.6 | 66.0 | 65.2 |
| **Open-Source General Models** |
| InternVL2.5-VL-8B | 46.8 | 35.0 | 50.0 | 50.8 | 45.6 |
| Qwen-2.5-VL-7B | 58.4 | 45.4 | 56.4 | 54.0 | 53.6 |
| InternVL2.5-VL-38B | 61.6 | 49.8 | 60.4 | 60.0 | 58.0 |
| Qwen-2.5-VL-32B | 71.6 | 59.4 | 69.6 | 66.6 | 66.8 |
| InternVL2.5-VL-78B | 59.8 | 53.2 | 68.0 | 65.2 | 61.6 |
| Qwen-2.5-VL-72B | 75.6 | 64.8 | 69.6 | 72.0 | 70.5 |
| **Open-Source Reasoning Models** |
| InternVL2.5-8B-MPO | 26.6 | 25.0 | 42.4 | 44.0 | 34.5 |
| InternVL2.5-38B-MPO | 41.4 | 42.8 | 55.8 | 53.2 | 48.3 |
| QVQ-72B-Preview | 61.4 | 57.4 | 62.6 | 64.4 | 61.5 |
| Adora | 63.6 | 50.6 | 59.0 | 59.0 | 58.1 |
| R1-Onevision | 44.8 | 33.8 | 39.8 | 40.8 | 39.8 |
| OpenVLThinker-7 | 63.0 | 53.8 | 60.6 | 65.0 | 60.6 |
| **Ours** |
| MM-Eureka-7B | 71.2 | 56.2 | 65.2 | 65.2 | 64.5 |
| MM-Eureka-32B | 74.6 | 62.0 | 75.4 | 76.8 | 72.2 |


## Data fields
| Key        | Description                         |
| ---------- | ----------------------------------- |
| `id`    | ID.                         |
| `subject`    | subject: math, physics, chemistry, and biology                         |
| `image`    | Image path.                         |
| `question` | Input query.                        |
| `answer`   | Verified Answer.   |

## Citation
If you find this project useful in your research, please consider citing:
```BibTeX
@article{meng2025mm,
  title={MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning},
  author={Meng, Fanqing and Du, Lingxiao and Liu, Zongkai and Zhou, Zhixiang and Lu, Quanfeng and Fu, Daocheng and Shi, Botian and Wang, Wenhai and He, Junjun and Zhang, Kaipeng and others},
  journal={arXiv preprint arXiv:2503.07365},
  year={2025}
}
```