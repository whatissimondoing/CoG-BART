import os
import torch
# import fitlog
import logging
import numpy as np

from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def train(train_dataloader, eval_dataloader, model, training_args, other_args):
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    best_score = 0

    steps_per_epoch = len(train_dataloader)

    # total number of training steps
    num_train_steps = int(steps_per_epoch * training_args.num_train_epochs)
    t_total = num_train_steps

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_ratio * t_total, num_training_steps=t_total)

    # multi-gpu training
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0

    for epoch in trange(int(training_args.num_train_epochs), desc="Epoch"):

        training_steps = 0
        model.zero_grad()

        for data in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            model.train()
            outputs = model(**data)
            loss, ce_loss, cl_loss, gen_loss = outputs.loss, outputs.ce_loss, outputs.cl_loss, outputs.gen_loss

            if training_args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            training_steps += 1
            global_step += 1

            if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:

                # fitlog.add_loss(loss, name="Loss", step=global_step)
                # fitlog.add_loss(ce_loss, name="CE_Loss", step=global_step)
                # fitlog.add_loss(cl_loss, name="CL_Loss", step=global_step)
                # fitlog.add_loss(gen_loss, name="Gen_Loss", step=global_step)

                results = evaluate(training_args, other_args, eval_dataloader, model, "evaluate")
                torch.cuda.empty_cache()
                res_for_display = {}
                for k, v in results.items():
                    res_for_display[k.replace("-", "_")] = v
                # fitlog.add_metric({"dev": res_for_display}, step=global_step)
                if other_args.task_name in ['MELD', 'IEMOCAP', 'EmoryNLP']:
                    eval_metrics = 'macro_f1'
                else:
                    eval_metrics = 'micro_f1'
                if results[eval_metrics] > best_score:
                    best_score = results[eval_metrics]
                    # fitlog.add_best_metric({"dev": {eval_metrics: best_score}})

                    # save the best model
                    output_dir = os.path.join(training_args.output_dir, "best_model_%d" % training_args.seed)
                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # results = evaluate(training_args, other_args, test_dataloader, model, "predict")
                    # fitlog.add_metric({"test": {'macro_f1': results['macro_f1']}}, step=global_step)
                    # fitlog.add_metric({"test": {'micro_f1': results['micro_f1']}}, step=global_step)
                    #
                    # fitlog.add_best_metric({"test": {'macro_f1': results['macro_f1']}})
                    # fitlog.add_best_metric({"test": {'micro_f1': results['micro_f1']}})

        torch.cuda.empty_cache()


def evaluate(training_args, other_args, eval_loader, model, eval_or_test):
    def compute_acc_for_categories(preds, labels):
        categories_count = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_right = {"label_%s" % i: 0 for i in range(other_args.num_labels)}
        categories_acc = {}
        for pred, label in zip(preds, labels):
            categories_count["label_%s" % label] += 1
            if pred == label:
                categories_right["label_%s" % label] += 1
        for index, (key, value) in enumerate(categories_count.items()):
            categories_acc["label_%s" % index] = format(categories_right["label_%s" % index] / value, '.4f')
        print(categories_acc)
        return categories_acc

    def compute_metrics(eval_preds):
        results = {}
        preds_id, labels_id, pred_text_id, gold_text_id = eval_preds

        # -------------- eval classification --------------
        accuracy = round(accuracy_score(labels_id, preds_id) * 100, 4)
        if other_args.task_name in ['MELD', 'EmoryNLP', 'IEMOCAP']:
            macro_f1 = f1_score(preds_id, labels_id, labels=[i for i in range(other_args.num_labels)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(other_args.num_labels)], average='micro')
        else:
            macro_f1 = f1_score(preds_id, labels_id, labels=[i for i in range(1, other_args.num_labels)], average='weighted')
            micro_f1 = f1_score(labels_id, preds_id, labels=[i for i in range(1, other_args.num_labels)], average='micro')
        results['acc'] = accuracy
        results['macro_f1'] = round(macro_f1 * 100, 4)
        results['micro_f1'] = round(micro_f1 * 100, 4)

        return results

    results = {}

    if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
        os.makedirs(training_args.output_dir)

    # training_args.eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu eval
    if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running %s *****" % eval_or_test)
    logger.info("  Num examples = %d", len(eval_loader.dataset))
    logger.info("  Batch size = %d", training_args.eval_batch_size)
    # eval_loss = 0.0

    all_preds, all_labels, all_pred_text, all_gold_text = [], [], [], []
    all_input_ids = []
    all_hidden_state = []

    max_input_len = 0
    for batch in tqdm(eval_loader, desc=eval_or_test):
        model.eval()
        batch = tuple(v.to(training_args.device) for _, v in batch.items())

        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'speakers': batch[3]}
            labels = batch[2]

            all_input_ids.append(batch[0].cpu().numpy())

            labels = labels.cpu().numpy()

            outputs = model(**inputs)
            preds = outputs.cls_logits
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
            all_labels.append(labels)
            all_preds.append(preds)

            max_input_len = max(max_input_len, inputs["input_ids"].shape[-1])

            hidden_state = outputs.last_hidden_states.cpu().numpy()
            all_hidden_state.append(hidden_state)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_pred_text, all_gold_text = None, None

    correct_num = np.sum(all_preds == all_labels)

    # eval_loss = eval_loss / nb_eval_steps
    result = compute_metrics((all_preds, all_labels, all_pred_text, all_gold_text))
    results.update(result)
    logger.info("***** %s results *****" % eval_or_test)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))
    logger.info("Correct / Total num = ({}/{})".format(correct_num, len(all_labels)))

    return results
