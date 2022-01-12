export TASK=$1

EPOCHS=3
TRAIN_BATCH_SIZE=2
LOGGING_STEPS=50
WARMUP_RATIO=0.1
LEARNING_RATE=2e-5
MODEL_SIZE=large
ALPHA=0.4
BETA=0.1
SEED=42

TRAIN_WITH_SPEAKER=1
TRAIN_WITH_GENERATION=1

python utils/data_process.py \
  --train_with_generation $TRAIN_WITH_GENERATION \
  --train_with_speaker $TRAIN_WITH_SPEAKER \
  --task_name $TASK

if [ $TASK = 'MELD' ]; then
  if [ $MODEL_SIZE = 'base' ]; then
    EPOCHS=12
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=40
    WARMUP_RATIO=0.3
    LEARNING_RATE=1e-5
    ALPHA=0.05
    BETA=0.05
  elif [ $MODEL_SIZE = 'large' ]; then
    EPOCHS=4
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=100
    WARMUP_RATIO=0.5
    LEARNING_RATE=1e-5
    ALPHA=0.4
    BETA=0.05
  fi
elif [ $TASK = 'EmoryNLP' ]; then
  if [ $MODEL_SIZE = 'base' ]; then
    EPOCHS=12
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=20
    WARMUP_RATIO=0.1
    LEARNING_RATE=2e-5
    ALPHA=0.2
    BETA=0.1
  elif [ $MODEL_SIZE = 'large' ]; then
    EPOCHS=10
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=80
    WARMUP_RATIO=0.4
    LEARNING_RATE=1e-5
    ALPHA=0.2
    BETA=0.1
  fi
elif [ $TASK = 'DailyDialog' ]; then
  if [ $MODEL_SIZE = 'base' ]; then
    EPOCHS=5
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=500
    WARMUP_RATIO=0.3
    LEARNING_RATE=2e-5
    ALPHA=0.01
    BETA=0.1
  elif [ $MODEL_SIZE = 'large' ]; then
    EPOCHS=4
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=500
    WARMUP_RATIO=0.3
    LEARNING_RATE=2e-5
    ALPHA=0.1
    BETA=0.1
  fi
elif [ $TASK = 'IEMOCAP' ]; then
  if [ $MODEL_SIZE = 'base' ]; then
    EPOCHS=20
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=20
    WARMUP_RATIO=0.1
    LEARNING_RATE=1.4e-5
    ALPHA=0.4
    BETA=0.1
  elif [ $MODEL_SIZE = 'large' ]; then
    EPOCHS=25
    TRAIN_BATCH_SIZE=2
    LOGGING_STEPS=80
    WARMUP_RATIO=0.4
    LEARNING_RATE=2e-5
    ALPHA=0.4
    BETA=0.1
  fi
fi

EVAL_BATCH_SIZE=$(expr 3 \* $TRAIN_BATCH_SIZE)

python main.py \
--model_name_or_path facebook/bart-$MODEL_SIZE \
--do_train \
--task_name $TASK \
--num_train_epochs $EPOCHS \
--learning_rate $LEARNING_RATE \
--output_dir ./save/$TASK \
--overwrite_output_dir \
--per_device_train_batch_size $TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $EVAL_BATCH_SIZE \
--logging_steps $LOGGING_STEPS \
--warmup_ratio $WARMUP_RATIO \
--adam_epsilon 1e-6 \
--weight_decay 0.01 \
--seed $SEED \
--alpha $ALPHA \
--beta $BETA \
--temperature 1 \
--use_trans_layer 1 \
--train_with_generation $TRAIN_WITH_GENERATION


