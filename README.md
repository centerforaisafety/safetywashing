# Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?

[![arXiv](https://img.shields.io/badge/arXiv-2407.21792-b31b1b.svg)](https://arxiv.org/abs/2407.21792)

This repository contains code and data for the paper ["Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?"](https://arxiv.org/abs/2407.21792).

"Safetywashing" refers to the practice of misrepresenting capabilities improvements as safety advancements in AI systems. This project provides tools to evaluate AI models on various safety and capabilities benchmarks, and analyzes the correlations between these benchmarks. Our goal is to empirically investigate whether common AI safety benchmarks actually measure distinct safety properties or are primarily determined by upstream model capabilities.

## Repository Structure

- `analysis.py`: Main script for running correlation analyses
- `data/`: Contains benchmark datasets and model results
  - `E_base_model.csv`: Results for base language models
  - `E_chat_model.csv`: Results for chat/instruction-tuned models
  - `evals_info.csv`: Metadata about evaluation benchmarks

## Citation

If you use this code or data in your research, please cite our paper:

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

Or for a nicer looking citation,

```
@article{ren2024safetywashing,
      title={Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?}, 
      author={Richard Ren and Steven Basart and Adam Khoja and Alice Gatti and Long Phan and Xuwang Yin and Mantas Mazeika and Alexander Pan and Gabriel Mukobi and Ryan H. Kim and Stephen Fitz and Dan Hendrycks},
      year={2024},
      journal={arXiv preprint arXiv: 2407.21792},
}
```
