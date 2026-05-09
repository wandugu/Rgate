import os
import random
from collections import defaultdict

import numpy as np
import torch
import constants
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB2
from config_utils import get_logger, get_logging_config


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(loader, model, optimizer, task, weight=1.0):
    losses = []
    logger = get_logger()

    model.train()
    logger.debug("开始训练任务=%s", task)
    for batch_index, batch in enumerate(tqdm(loader), start=1):
        optimizer.zero_grad()
        loss, _ = getattr(model, f'{task}_forward')(batch)
        loss *= weight
        loss.backward()
        optimizer.step()
        if hasattr(model, "update_ema"):
            model.update_ema()
        losses.append(loss.item())
        logger.debug("训练任务=%s, batch=%s, loss=%.6f", task, batch_index, loss.item())

    logger.debug("训练任务=%s完成, 平均loss=%.6f", task, np.mean(losses))
    return np.mean(losses)


def log_prediction_samples(tokens, true_labels, pred_labels, total_samples=None):
    logger = get_logger()
    logging_config = get_logging_config()
    sample_template = logging_config["sample_template"]
    if total_samples is None:
        total_samples = len(tokens)
    for index, (token_list, true_label, pred_label) in enumerate(
        zip(tokens, true_labels, pred_labels),
        start=1,
    ):
        input_text = " ".join(token_list)
        message = sample_template.format(
            index=index,
            total=total_samples,
            input=input_text,
            true=true_label,
            pred=pred_label,
        )
        logger.info(message)


def evaluate(model, loader, return_preds: bool = False, log_samples: bool = False):
    true_labels = []
    pred_labels = []
    tokens = []
    logger = get_logger()
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None

    model.eval()
    with torch.no_grad():
        logger.debug("开始评估, 预估样本数=%s", total_samples)
        for batch_index, batch in enumerate(tqdm(loader), start=1):
            _, pred = model.ner_forward(batch)
            pairs = batch["pairs"] if isinstance(batch, dict) else batch
            tokens += [[token.text for token in pair.sentence] for pair in pairs]
            true_labels += [[constants.ID_TO_LABEL[token.label] for token in pair.sentence] for pair in pairs]
            pred_labels += pred
            logger.debug(
                "评估batch=%s, batch_size=%s",
                batch_index,
                len(pairs),
            )

    total = sum(len(seq) for seq in true_labels)
    correct = sum(
        t == p
        for seq_t, seq_p in zip(true_labels, pred_labels)
        for t, p in zip(seq_t, seq_p)
    )
    wrong = total - correct

    entity_correct_counts = defaultdict(int)
    entity_total_counts = defaultdict(int)

    for seq_t, seq_p in zip(true_labels, pred_labels):
        true_entities = set(get_entities(seq_t))
        pred_entities = set(get_entities(seq_p))

        for entity in true_entities:
            entity_total_counts[entity[0]] += 1

        for entity in true_entities & pred_entities:
            entity_correct_counts[entity[0]] += 1

    f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
    report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2)
    logger.debug(
        "评估完成: f1=%.6f, total=%s, correct=%s, wrong=%s",
        f1,
        total,
        correct,
        wrong,
    )

    if log_samples:
        log_prediction_samples(tokens, true_labels, pred_labels, total_samples=total_samples)

    if return_preds:
        return (
            f1,
            report,
            total,
            correct,
            wrong,
            dict(entity_correct_counts),
            dict(entity_total_counts),
            tokens,
            pred_labels,
            true_labels,
        )

    return f1, report, total, correct, wrong, dict(entity_correct_counts), dict(entity_total_counts)
