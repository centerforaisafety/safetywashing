# \[Development Branch: Adding code to replicate all evaluations\] Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?

[![arXiv](https://img.shields.io/badge/arXiv-2407.21792-b31b1b.svg)](https://arxiv.org/abs/2407.21792)

This repository contains code and data for the paper ["Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?"](https://arxiv.org/abs/2407.21792).

"Safetywashing" refers to the practice of misrepresenting capabilities improvements as safety advancements in AI systems. This project provides tools to evaluate AI models on various safety and capabilities benchmarks, and analyzes the correlations between these benchmarks. In our paper, we empirically investigate whether common AI safety benchmarks actually measure distinct safety properties or are primarily determined by upstream model capabilities.

## Repository Structure

- `analysis.py`: Main script for running correlation analyses
- `data/`: Contains benchmark datasets and model results
  - `benchmarks_base_models.csv`: Benchmark matrix for base language models
  - `benchmarks_chat_models.csv`: benchmark matrix for chat/instruction-tuned models
  - `benchmarks_info.csv`: Metadata about benchmarks

To add a new benchmark, add a column in the benchmark matrix, and then add a row in benchmarks_info to add metadata.

## Citation

```
@misc{ren2024safetywashing,
      title={Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?}, 
      author={Richard Ren and Steven Basart and Adam Khoja and Alice Gatti and Long Phan and Xuwang Yin and Mantas Mazeika and Alexander Pan and Gabriel Mukobi and Ryan H. Kim and Stephen Fitz and Dan Hendrycks},
      year={2024},
      eprint={2407.21792},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.21792}, 
}
```
