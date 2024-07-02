# Visualizing Dialogues: Enhancing Image Selection through Dialogue Understanding with Large Language Models

## Setup

```bash
conda create -n {your env name} python=3.12.2
conda activate {your env name}
pip install -r requirements.txt

# setup openai-clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Data Pre-processing

### PhotoChat

```bash
# put the raw photochat dataset in your directory
python utils/data/raw_data_processor.py --raw_path {raw photochat dataset directory} --saved_path {processed data directory}
```

## Run

### Zero-shot

```bash
# Generate descriptors
python3 utils/data/llm_inference.py --src_path {data directory} --saved_path {descriptor file saved directory} --model_name {LLM model name} --task {descriptor type: query, guess, sum}

# Run zero-shot
python3 CLDiagDescriptor.py --task {choose your target task} --src_path {descriptor file saved directory} --clip_model_name {CLIP model name} --zero_shot
```

### Fully-trained

```bash
# Generate descriptors
python3 utils/data/llm_inference.py --src_path {data directory} --saved_path {descriptor file saved directory} --model_name {LLM model name} --task {descriptor type: query, guess, sum}

# Run fully-trained
python3 CLDiagDescriptor.py --task {choose your target task} --src_path {descriptor file saved directory} --clip_model_name {CLIP model name} --n_epochs {number of epochs}
```
