# CoG-BART

[Contrast and Generation Make BART a Good Dialogue Emotion Recognizer](https://arxiv.org/abs/2112.11202)

[//]: # (>Bug report:)

[//]: # (> )

[//]: # (> &#40;Fixed&#41; There was a bug found in the project we referenced &#40;see [something678/TodKat#10]&#40;https://github.com/something678/TodKat/issues/10&#41;&#41;, so it was inherited as well. We have fixed the bug and the new result will be reported later.)

[//]: # (>> The new results have been reported in the new version of this paper.)

## Required Packages:

* torch==1.7.1
* transformers==4.11.0
* numpy
* pickle
* tqdm
* sklearn
* fitlog

## Run on GPU:

Model runs on one GPU by default, and we didn't try it on CPU.

> We recommend using GPU with memory more than 24G , otherwise you may need to adjust the hyperparameters and the results may vary significantly.

## Quick Start

To run the model on test sets of four datasets,

1. Download the pre-trained models:

   For MELD, download the [checkpoint](https://www.dropbox.com/s/a86apfo82aenltj/best_model_762985.zip?dl=0) , unzip it to `./save/MELD`

   For EmoryNLP, download the [checkpoint](https://www.dropbox.com/s/1jvq9dhiwq481kg/best_model_468186.zip?dl=0) , unzip it to `./save/EmoryNLP`

   For IEMOCAP, download the [checkpoint](https://www.dropbox.com/s/fcaiu2twlmx4xdc/best_model_981929.zip?dl=0) , unzip it to `./save/IEMOCAP`

   For DailyDialog, download the [checkpoint](https://www.dropbox.com/s/o4n14dmr3lh9lzd/best_model_738497.zip?dl=0) , unzip it to `./save/DailyDialog`

3. Execute the following command in terminal:

   For MELD: `bash eval.sh MELD ./save/MELD/best_model_762985`

   For EmoryNLP: `bash eval.sh EmoryNLP ./save/EmoryNLP/best_model_468186`

   For IEMOCAP: `bash eval.sh IEMOCAP ./save/IEMOCAP/best_model_981929`

   For DailyDialog: `bash eval.sh DailyDialog ./save/DailyDialog/best_model_738497`

## Training:

For MELD: `bash train.sh MELD`

For EmoryNLP: `bash train.sh EmoryNLP`

For IEMOCAP: `bash train.sh IEMOCAP`

For DailyDialog: `bash train.sh DailyDialog`

> It should be noticed that performance is greatly affected by random seed. So we recommended some seeds in the script for reproduction.

## Evaluation and Prediction:

For MELD: `bash eval.sh MELD save/MELD/best_model_939239`

For EmoryNLP: `bash eval.sh EmoryNLP save/EmoryNLP/best_model_552848`

For IEMOCAP: `bash eval.sh IEMOCAP save/IEMOCAP/best_model_625968`

For DailyDialog: `bash eval.sh DailyDialog save/DailyDialog/best_model_269130`

## Citation

If you find this work useful, please cite our work:

```
@misc{li2021contrast,
    title={Contrast and Generation Make BART a Good Dialogue Emotion Recognizer}, 
    author={Shimin Li and Hang Yan and Xipeng Qiu},
    year={2021},
    eprint={2112.11202},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgement

Some code of this project are referenced from [TodKat](https://github.com/something678/TodKat)
and [DialogXL](https://github.com/shenwzh3/DialogXL). We thank their open source materials for contribution to this task.

