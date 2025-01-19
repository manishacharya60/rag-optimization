# AUTOPATCH: Context-Aware Code Optimization through Retrieval-Augmented Generation

This repository contains the implementation of **AUTOPATCH**, a framework designed to optimize program runtime performance through **Context-Aware Retrieval-Augmented Generation (RAG)**. By combining insights from Control Flow Graph (CFG) analysis, retrieval-augmented learning, and in-context LLM prompting, AUTOPATCH bridges the gap between traditional manual code optimization and automated techniques. It achieves measurable improvements in execution efficiency while maintaining high adaptability to various programming challenges.

---

## Key Features

- **CFG-Based Optimization**: Leverages Control Flow Graph analysis to identify inefficiencies in code structure and guide optimizations.
- **Context-Aware Learning**: Integrates historical examples and optimization patterns through a retrieval-augmented pipeline.
- **Unified RAG Framework**: Embeds CFG differences and optimization rationales into structured prompts for precise and effective code refinement.
- **Comprehensive Evaluation**: Tested on IBM Project CodeNet, demonstrating a **7.3% improvement** in execution efficiency over baseline methods.
- **Modular Design**: Structured for easy experimentation, scalability, and integration with additional datasets and optimization techniques.

---

## Repository Structure

```plaintext
project_root/
├── cfg_conversion/
│   └── cfg_conversion.py   # Script for CFG generation and analysis
├── data/
│   ├── cfg_dataset/
│   │   ├── train_cfg_dataset.json   # Training dataset
│   │   └── test_cfg_dataset.json    # Testing dataset
│   ├── generated_code/
│   │   ├── context_generation.json # Optimized code using contextual examples
│   │   ├── naive_generation.json   # Optimized code using naive retrieval
│   │   ├── zero_shot_generation.json # Optimized code without context
│   │   ├── code_analysis.json      # Analysis results
│   │   ├── train_dataset.json      # Original training dataset
│   │   └── test_dataset.json       # Original testing dataset
├── embeddings/
│   ├── embeddings_script.py        # Script for generating code embeddings
│   ├── train_embeddings.csv        # Embeddings for training data
│   └── test_embeddings.csv         # Embeddings for testing data
├── evaluation/
│   └── analysis.ipynb              # Jupyter notebook for evaluation and visualization
├── .env                            # Environment configuration
├── .gitignore                      # Git ignore file
├── code_analysis.py                # Code analysis pipeline
├── context_aware.py                # Context-aware optimization script
├── naive_embeddings.py             # Embedding-based naive retrieval script
├── zero_shot.py                    # Zero-shot optimization script
└── requirements.txt                # List of required Python packages
