# MyEgo

## Introduction
Recent **multimodal large language models (MLLMs)** have made significant progress in egocentric video understanding and have been deployed on devices such as smart glasses. 

However, few works have explored whether MLLMs can perform ego-grounding - the ability to both understand and memorize **personalized concepts** (e.g., identifying and recalling which objects or actions 'belong' to the camera wearer in a video). 

**MyEgo** fills this gap. It is the first egocentric video QA dataset designed to evaluate MLLMs' ability to understand, remember, and reason about the camera wearer.

## Overview
- 541 videos collected from 3 egocentric video dataset.
- 5,012 manually annotated questions, each with open-ended (OE) and multiple-choice (MC) subtasks.
  - **Personalized**: highlighting distinctions between the camera wearer's (*my*) actions or objects and those of others, compelling the model to engage in personalized reasoning to first determine "which one is mine?" before arriving at a correct answer.
- Highly challenging, even top models such as GPT-5 achieve only 46% accuracy, significantly falling behind human performance (85%).

**Datset Statistics**

![alt text](figures/statstic.png)

**Comparion to existing egocentric video QA dataset**:

![alt text](figures/comparison.png)
## Dataset access
1. **Download the videos:** Please download our video source from [here](https://drive.google.com/drive/u/1/folders/1rZo-6X_Xst_9J9TzJOJ1owW3ZWUstOMl).
2. **Video preprocessing:** You shall need to preprocess the videos collected from [Egolife](https://github.com/EvolvingLMMs-Lab/EgoLife) to remove the timestamp watermark.
```bash
python scripts/mask.py --input_folder <path_to_videos> --output_folder <path_to_save_unmarked_videos>
```
3. **Obtain QA files:** Please see `data/myego.json` for the QA files.

## Evaluation Result
<details><summary>Evaluation results of both open-source and closed-source MLLMs on MyEgo</summary>

![alt text](figures/result.png)
</details>
<details><summary>Visualization of the evaluation results</summary>

![alt text](figures/case.png)

![alt text](figures/morecase.png)
</details>



## Evaluation Pipeline
To be released soon.

## License
For the video sources, please refer to the original dataset licenses: 
- [Ego4D](https://ego4ddataset.com/ego4d-license/)
- [CASTLE](https://castle-dataset.github.io)
- [Egolife](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/main/LICENSE)
