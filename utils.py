import os
import random
import numpy as np
import torch
import constants
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2


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

    model.train()
    for batch in tqdm(loader):
        optimizer.zero_grad()
        loss, _ = getattr(model, f'{task}_forward')(batch)
        loss *= weight
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(model, loader, return_preds: bool = False):
    true_labels = []
    pred_labels = []
    tokens = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            _, pred = model.ner_forward(batch)
            pairs = batch["pairs"] if isinstance(batch, dict) else batch
            tokens += [[token.text for token in pair.sentence] for pair in pairs]
            true_labels += [[constants.ID_TO_LABEL[token.label] for token in pair.sentence] for pair in pairs]
            pred_labels += pred

    total = sum(len(seq) for seq in true_labels)
    correct = sum(
        t == p
        for seq_t, seq_p in zip(true_labels, pred_labels)
        for t, p in zip(seq_t, seq_p)
    )
    wrong = total - correct

    f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
    report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2)

    if return_preds:
        return f1, report, total, correct, wrong, tokens, pred_labels, true_labels

    return f1, report, total, correct, wrong
