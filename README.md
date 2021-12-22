# CoG-BART

[Contrast and Generation Make BART a Good Dialogue Emotion Recognizer](https://arxiv.org/abs/2112.11202)


## Quick Start:
------------------------------------------------------
To run the model on test sets of four datasets,

1. Download the pre-trained models:

   * For MELD:
       download the checkpoint: [best_model_939239.tar](https://www.dropbox.com/s/c6tfmy9vuxtqpx4/best_model_939239.tar?dl=0) ,unzip the file to ./save/MELD
   
   * For IEMOCAP:
       download the checkpoint: [best_model_625968.tar](https://www.dropbox.com/s/btc68s239zcf9dj/best_model_625968.tar?dl=0) ,unzip the file to ./save/IEMOCAP
   
   * For EmoryNLP:
       download the checkpoint: [best_model_552848.tar](https://www.dropbox.com/s/dcucsdee8hmhu0o/best_model_552848.tar?dl=0) ,unzip the file to ./save/EmoryNLP
   
   * For DailyDialog:
       download the checkpoint: [best_model_269130.tar](https://www.dropbox.com/s/rgn6o2obbaz8vh4/best_model_269130.tar?dl=0) ,unzip the file to ./save/DailyDialog

2. Execute the following command in terminal:

    * For MELD: `bash eval.sh MELD save/MELD/best_model_939239`

    * For EmoryNLP: `bash eval.sh EmoryNLP save/EmoryNLP/best_model_552848`

    * For IEMOCAP: `bash eval.sh IEMOCAP save/IEMOCAP/best_model_625968`

    * For DailyDialog: `bash eval.sh DailyDialog save/DailyDialog/best_model_269130`


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

