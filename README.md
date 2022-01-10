# CoG-BART

[Contrast and Generation Make BART a Good Dialogue Emotion Recognizer](https://arxiv.org/abs/2112.11202)

>Bug report:
> 
> There was a bug found in the project we referenced (see [something678/TodKat#10](https://github.com/something678/TodKat/issues/10)), so it was inherited as well. We have fixed the bug and the new result will be reported later.



## Required Packages:
------------------------------------------------------

* torch==1.7.1
* transformers==4.11.0
* numpy
* pickle
* tqdm
* sklearn
* fitlog

## Run on GPU:
------------------------------------------------------
Model runs on one GPU by default, and we didn't try it on CPU.

> We recommend using GPU with memory more than 24G , otherwise you may need to adjust the hyperparameters and the results may vary significantly.

## Training:
------------------------------------------------------
For MELD: `bash train.sh MELD`

For EmoryNLP: `bash train.sh EmoryNLP`

For IEMOCAP: `bash train.sh IEMOCAP`

For DailyDialog: `bash train.sh DailyDialog`

>It should be noticed that performance is greatly affected by random seed. So we recommended some seed in the script for reproduction.

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

Some code of this project is referenced from [TodKat](https://github.com/something678/TodKat)
and [DialogXL](https://github.com/shenwzh3/DialogXL). We thank their open source materials for contribution to this task.

