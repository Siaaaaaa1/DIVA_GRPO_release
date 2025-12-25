# Multimodal Reasoning & Variation Synthesis

This module provides a streamlined pipeline for augmenting multimodal datasets using Large Language Models (LLMs). It automates the generation of **Problem Variants** and **Chain-of-Thought (CoT) Reasoning Steps** to enhance dataset diversity and difficulty.

## 📂 Core Components

The pipeline is built around two clean, decoupled files designed to replace legacy scripts:

**`augment_dataset.py`**
* **Role**: The main entry point (Driver Script).
* **Functionality**: Handles data loading (Parquet), multiprocessing dispatch, error handling, and incremental saving. It ensures that large datasets are processed efficiently without data loss.


**`llm_client.py`**
* **Role**: The logic & API layer.
* **Functionality**: Encapsulates all interactions with the Azure OpenAI/Qwen API. It manages prompt engineering, image encoding (Base64), and robust parsing of XML-tagged model outputs (e.g., `<step1>`, `<variant1>`).



## ⚙️ Setup

### 1. Dependencies

Ensure your environment supports the following:

```bash
pip install pandas pyarrow openai tenacity tqdm

```

### 2. Environment Variables

You should set your Azure OpenAI credentials. Alternatively, you can pass them as command-line arguments.

```bash
export AZURE_OPENAI_KEY="your-api-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

```

## 🚀 Usage

Run the `augment_dataset.py` script to process your `.parquet` dataset.

### Basic Command

```bash
python augment_dataset.py \
  --input "datasets/train_data.parquet" \
  --output "datasets/train_data_augmented.parquet" \
  --workers 8

```

## 📊 Output Data Structure

The script generates a new Parquet file containing the original columns plus the following new fields:

| Column Name | Type | Description |
| --- | --- | --- |
| `variants` | `List[str]` | A list of 5 rephrased versions of the original problem (preserving the correct answer). |
| `think_steps` | `List[str]` | Step-by-step reasoning traces (CoT) generated and refined by the model. |
| `think_answer` | `str` | The final answer extracted from the reasoning process (used for consistency checks). |