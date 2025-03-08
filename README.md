# Treble-Counterfactual-VLMs
The repository follows the directory structure below:

```
Treble-Counterfactual-VLMs/
├── experiments/
├── lavis/
├── llava/
├── TTI_utils/
├── POPE/
└── README.md
```

## Setup Instructions

Before running the project, please download the required dependencies:

- **LAVIS**: Clone from [Salesforce LAVIS](https://github.com/salesforce/LAVIS/lavis) and place it in the `lavis/` directory.
- **LLaVA**: Clone from [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main/llava) and place it in the `llava/` directory.
- **POPE**: Clone from [POPE](https://github.com/RUCAIBox/POPE) and place it in the `POPE/` directory.
- Download the COCO **train2014** and **val2014** datasets, and place them in the following directories:

```
experiments/data/train2014/
experiments/data/val2014/
```
