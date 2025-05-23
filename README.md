# CoMEM: Towards General Continuous Memory for Vision-Language Models

This is the official code repository for the paper: Towards General Continuous Memory for Vision-Language Models.

<img width="1728" alt="image" src="https://github.com/WenyiWU0111/CoMEM/blob/main/images/framework.jpg">

CoMEM introduces a novel approach for integrating multimodal and multilingual knowledge into Vision-Language Models (VLMs) using a compact and efficient continuous memory—a small set of dense embeddings. Unlike traditional retrieval-based methods, CoMEM leverages the VLM itself as a memory encoder, fine-tuned with just **1.2%** of the model’s parameters and a lightweight corpus of **15.6K** self-synthesized samples. This design enables CoMEM to encode arbitrary knowledge into only 8 continuous embeddings, improving performance on complex multimodal reasoning tasks. Importantly, the base VLM remains frozen at inference, making our memory module **plug-and-play** and easily adaptable across tasks and domains.

## 📦 Requirements
To get started, please create a new environment and install the required dependencies:

```bash
conda create -n CoMEM python=3.10
conda activate CoMEM
pip install -r requirements.txt
```

## 📚 Retrival Data Base
Our knowledge base is constructed using the [Wikipedia-based Image-Text (WIT)](https://github.com/google-research-datasets/wit) dataset. We provide a pre-built FAISS index of CLIP embeddings over WIT. You can use the provided scripts to reconstruct the database using this index.

## 🔥 Training

The Memory Encoder in CoMEM is trained on a mixture of datasets, including: [Infoseek](https://github.com/edchengg/infoseek_eval), [EVQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa), [OKVQA](https://okvqa.allenai.org/), and multi-lingual Infoseek. Each training sample is paired with the top-3 most relevant retrieved image-text pairs. Download our training data [here](link).

To train the Memory Encoder, run the appropriate script below based on the model configuration:

- For CoMEM with Qwen2-VL-Instruct or Qwen2.5-VL-Instruct, use:
```bash
  bash scripts/train/finetune_lora_vision_vlm.sh
```

- For CoMEM with Qwen2.5-Instruct, use:
```bash
bash scripts/train/finetune_lora_vision_llm.sh
```

After training, remember to merge the LoRA checkpoint with the base model using the following script:
```bash
bash scripts/train/merge_lora.sh
```

**Note:** Due to instability issues with ```transformers==4.49.0```, we have made a minor modification to ```src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py```. If you encounter errors related to Qwen2.5-VL when using this version of Transformers, please replace the file with our patched version available [here](https://github.com/WenyiWU0111/CoMEM/blob/main/patch/modeling_qwen2_5_vl.py).


## 🤗 Checkpoints
We release our checkpoints to Huggingface. You can access them through the following links:
- [CoMEM with Qwen2.5-VL(VLM)](https://huggingface.co/WenyiWU0111/continuous-memory-qwen2.5-800)
- [CoMEM with Qwen2-VL(VLM)](https://huggingface.co/WenyiWU0111/continuous-memory-qwen2-800)
- [CoMEM with Qwen2.5-Instruct(LLM)](https://huggingface.co/WenyiWU0111/continuous-memory-qwen2.5-llm)
## 🔍 Inference

We provide scripts for running inference with baseline, vanilla RAG, and CoMEM settings across **8** benchmarks and **10+** models. Refer to the respective bash files for detailed usage.

- Baseline (Multimodal VQA):
```bash
bash scripts/inference/run_baseline_inference.sh
```

- Vanilla RAG (Multimodal VQA):
```bash
bash scripts/inference/run_rag_inference.sh
```

- CoMEM (Multimodal VQA):
```bash
bash scripts/inference/run_CoMEM_inference.sh
```

- Multilingual Multimodal VQA:
```bash
bash scripts/inference/run_multilingual_baseline_inference.sh
bash scripts/inference/run_multilingual_rag_inference.sh
bash scripts/inference/run_multilingual_CoMEM_inference.sh
```

## 📊 Evaluation

Please use the following scripts for evaluating performance on each benchmark:

- [AOK-VQA](https://github.com/allenai/aokvqa): CoMEM-inference/AOK-VQA/aokvqa_eval.py
- [OKVQA](https://okvqa.allenai.org/): CoMEM-inference/OK-VQA/okvqa_eval.py
- [ViQUAE](https://github.com/PaulLerner/ViQuAE): CoMEM-inference/Viquae/viquae_eval.py
- [Infoseek](https://github.com/edchengg/infoseek_eval): CoMEM-inference/infoseek/run_evaluation_rulebase.py
- [OVEN](https://github.com/edchengg/oven_eval): CoMEM-inference/OVEN/run_oven_eval.py
- [MRAG-Bench](https://mragbench.github.io/): CoMEM-inference/MRAG_Bench/mrag_bench_eval.py
- [CVQA](https://huggingface.co/datasets/afaji/cvqa): CoMEM-inference/CVQA/cvqa_eval.py

Note: For the OVEN benchmark, be sure to run the following first:
```bash
python CoMEM-inference/OVEN/run_bm25_index.py
python CoMEM-inference/OVEN/run_bm25_query.py
```
For more details, please refer to the official repo for [OVEN](https://github.com/edchengg/oven_eval).
