# MyEgo

## Overview
- 541 videos collected from 3 egocentric video dataset.
- 5,012 manually annotated questions, each with open-ended (OE) and multiple-choice (MC) subtasks.
  - **Personalized**: highlighting distinctions between the camera wearer's (*my*) actions or objects and those of others, compelling the model to engage in personalized reasoning to first determine "which one is mine?" before arriving at a correct answer.
- Highly challenging, even top models such as GPT-5 achieve only 46% accuracy, significantly falling behind human performance (85%).

**Datset Statistics**


![alt text](figures/stat_fin.png)

**Comparion to existing egocentric video QA dataset**:

![alt text](figures/comparison.png)
## Dataset access
1. **Download the videos:** You can download the videos from [here](https://drive.google.com/drive/u/0/folders/1rZo-6X_Xst_9J9TzJOJ1owW3ZWUstOMl). Our videos are sourced from [Ego4D](https://ego4d-data.org/docs/data/videos), [CASTLE](https://castle-dataset.github.io/#download) and [Egolife](https://huggingface.co/datasets/lmms-lab/EgoLife).

2. **Video preprocessing:** You shall need to preprocess the videos collected from **Egolife** to remove the timestamp watermark.
```bash
python mask.py --input_folder <path_to_videos> --output_folder <path_to_save_unmasked_videos>
```
3. **Obtain QA files:** Please see `data/myego.json` for the QA files.

## Evaluation Result
- Evaluation results of both open-source and closed-source MLLMs on MyEgo

![alt text](figures/result.png)

- Visualization of the evaluation results

![alt text](figures/case.png)

![alt text](figures/morecase.png)


## Evaluation Pipeline
- Inference

Inference with one GPU and batch size of 1 by default.

```bash
bash scripts/infer.sh
```

- GPT-based evaluation

```bash
python eval.py --api_key <your_api_key>
```
  

## License
For the video sources, please refer to the original dataset licenses: 
- [Ego4D](https://ego4ddataset.com/ego4d-license/)
- [CASTLE](https://castle-dataset.github.io)
- [Egolife](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/main/LICENSE)
