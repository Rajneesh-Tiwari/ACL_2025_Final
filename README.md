# ACL_2025_Final
Code for ACL 2025 Track 3 Workshop

# LLM Self-Reasoning Analysis Pipeline

This repository contains a system for fine-grained analysis of chain-of-thought (CoT) reasoning in Large Language Models, as detailed in our paper for the LLM-SR competition.

## Overview

Our approach implements a three-stage pipeline for analyzing and evaluating the quality of reasoning processes:

1. **Question Parsing (QP)**: Extracts and formalizes logical conditions from problem statements
2. **Chain-of-Thought Parsing (CP)**: Identifies key statements in reasoning attempts and their supporting evidence
3. **Quality Control (QC)**: Verifies and corrects outputs from both stages, ensuring reliability

The system uses the Llama-3-8B-Instruct model with carefully crafted prompts that encourage critical evaluation of reasoning steps.

## Requirements

```
unsloth==2025.3.9
vllm==0.7.3
tqdm
```

The code will attempt to automatically install these dependencies if they're not present.

## Usage

```
python inference.py --input_file path/to/input.json --output_prefix results
```

### Command-line Arguments

- `--input_file`: Path to input JSON file (required)
- `--output_prefix`: Prefix for output file names (default: 'results')
- `--save_every`: How often to save partial results (default: 10 samples)
- `--use_icl`: Use in-context learning templates (default: True)
- `--num_examples`: Number of examples to include in QC prompts (default: 2)
- `--debug`: Print debug information (default: False)

## Input/Output Format

### Input Format

The input file should be a JSON array of objects with the following structure:

```json
[
  {
    "id": 123,
    "question": "Problem statement...",
    "cot": "Chain of thought reasoning..."
  },
  ...
]
```

### Output Format

The system generates three output files:

1. `{prefix}_inference.json`: Results after initial question parsing and CoT parsing
2. `{prefix}_cp_qc.json`: Results after chain-of-thought quality control
3. `{prefix}_final.json`: Final results after question parsing quality control

The final output format extends the input with:

```json
{
  "id": 123,
  "question": "Problem statement...",
  "cot": "Chain of thought reasoning...",
  "question_parsing": [
    "condition 1",
    "condition 2",
    ...
  ],
  "cot_parsing": [
    {
      "statement": "direct quote of a key reasoning step",
      "evidence": "direct quote of the specific text the statement is based on",
      "Verification": "True or False"
    },
    ...
  ]
}
```

## Key Features

- **Systematic Verification**: Applies strict standards of logical validity to reasoning steps
- **In-Context Learning**: Uses embedded examples for few-shot learning
- **Error Resilience**: Implements robust retry mechanisms with adaptive temperature adjustments
- **Quality Control**: Provides additional verification layers for both condition extraction and reasoning analysis

## Pipeline Details

1. **Question Parsing**:
   - Extracts logical conditions from natural language problem descriptions
   - Uses template-based prompting with optional in-context learning

2. **Chain-of-Thought Parsing**:
   - Identifies 4-6 key statements from reasoning attempts
   - Extracts specific evidence for each statement
   - Verifies logical support using strict validity standards

3. **Quality Control Mechanisms**:
   - Verifies completeness and accuracy of condition extraction
   - Corrects verification judgments that may be initially too lenient
   - Provides a comprehensive review of statement-evidence pairs


## License

[MIT License](LICENSE)
