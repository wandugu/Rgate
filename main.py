import os
import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import loader
from data.dataset import collate_fn
from model.model import MyModel
from utils import seed_worker, seed_everything, train, evaluate
import constants
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def print_entity_statistics(entity_correct_counts, entity_total_counts):
    entity_types = sorted({label.split('-', 1)[1] for label in constants.ID_TO_LABEL if '-' in label} |
                          set(entity_correct_counts.keys()) |
                          set(entity_total_counts.keys()))

    if not entity_types:
        print('各实体类别正确数量与总数量：无实体样本')
        print('实体类别 Accuracy：无实体样本')
        return

    print('各实体类别正确数量与总数量：')
    for entity_type in entity_types:
        total = entity_total_counts.get(entity_type, 0)
        correct = entity_correct_counts.get(entity_type, 0)
        print(f'{entity_type}: 正确 {correct} 个，总数 {total} 个')

    print('实体类别 Accuracy：')
    for entity_type in entity_types:
        total = entity_total_counts.get(entity_type, 0)
        correct = entity_correct_counts.get(entity_type, 0)
        if total > 0:
            accuracy = correct / total
            print(f'实体类别 {entity_type} Accuracy= {correct}/{total} = {accuracy:.4f}')
        else:
            print(f'实体类别 {entity_type} Accuracy= 无测试样本')

    total_correct_values = [entity_correct_counts.get(entity_type, 0) for entity_type in entity_types]
    total_total_values = [entity_total_counts.get(entity_type, 0) for entity_type in entity_types]
    total_correct_sum = sum(total_correct_values)
    total_total_sum = sum(total_total_values)

    if total_total_sum > 0:
        correct_expr = '+'.join(str(value) for value in total_correct_values)
        total_expr = '+'.join(str(value) for value in total_total_values)
        overall_accuracy = total_correct_sum / total_total_sum
        print(
            f'实体判断Accuracy的打印 = '
            f'{{成功{{{correct_expr}={total_correct_sum}}}}}/'
            f'{{总数{total_expr}={total_total_sum}}} = {overall_accuracy:.4f}'
        )
    else:
        print('实体判断Accuracy的打印 = 无测试样本')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='twitter2017', choices=['twitter2015', 'twitter2017'])
    parser.add_argument('--encoder_t', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--encoder_v', type=str, default='', choices=['', 'resnet101', 'resnet152'])
    parser.add_argument('--stacked', action='store_true', default=False)
    parser.add_argument('--rnn',   action='store_true',  default=False)
    parser.add_argument('--crf',   action='store_true',  default=False)
    parser.add_argument('--aux',   action='store_true',  default=False)
    parser.add_argument('--gate',   action
    ='store_true',  default=False)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--save_interval', type=int, default=0,
                        help='save model every n epochs, 0 to disable')
    parser.add_argument('--load_model', type=str, default='',
                        help='path to a saved model for evaluation')
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if (args.aux or args.gate) and args.encoder_v == '':
        raise ValueError('Invalid setting: auxiliary task or gate module must be used with visual encoder (i.e. ResNet)')

    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    ner_corpus = loader.load_ner_corpus(f'datasets/{args.dataset}', load_image=(args.encoder_v != ''))
    ner_train_loader = DataLoader(ner_corpus.train, batch_size=args.bs, collate_fn=collate_fn, num_workers=args.num_workers,
                                  shuffle=True, worker_init_fn=seed_worker, generator=generator)
    ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=args.bs, collate_fn=collate_fn, num_workers=args.num_workers)
    ner_test_loader = DataLoader(ner_corpus.test, batch_size=args.bs, collate_fn=collate_fn, num_workers=args.num_workers)

    if args.aux:
        itr_corpus = loader.load_itr_corpus('datasets/relationship')
        itr_train_loader = DataLoader(itr_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                                      shuffle=True, worker_init_fn=seed_worker, generator=generator)
        itr_test_loader = DataLoader(itr_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)

    model = MyModel.from_pretrained(args)

    def save_result(mode, tokens, preds, trues):
        result_dir = 'result'
        os.makedirs(result_dir, exist_ok=True)
        gt_path = os.path.join(result_dir, f'{mode}_groundtruth.json')
        res_path = os.path.join(result_dir, f'{mode}_result.json')
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump([
                {'tokens': t, 'labels': l}
                for t, l in zip(tokens, trues)
            ], f, ensure_ascii=False, indent=4)
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump([
                {'tokens': t, 'labels': l}
                for t, l in zip(tokens, preds)
            ], f, ensure_ascii=False, indent=4)
        if mode == 'train':
            print(f'训练集文件输出至{gt_path}目录下')
        elif mode == 'test':
            print(f'测试集文件输出至{gt_path}目录下')
        else:
            print(f'{mode}集文件输出至{gt_path}目录下')
        print(f'模型预测文件输出至{res_path}目录下')

    if args.load_model:
        state = torch.load(args.load_model, map_location=model.device)
        model.load_state_dict(state)
        (
            test_f1,
            test_report,
            test_total,
            test_correct,
            test_wrong,
            test_entity_correct_counts,
            test_entity_total_counts,
            tokens,
            preds,
            trues,
        ) = evaluate(model, ner_test_loader, return_preds=True)
        print(f'f1 score on test set: {test_f1:.4f}')
        print(f'测试集共 {test_total} 个 token，预测正确 {test_correct} 个，预测错误 {test_wrong} 个')
        print(f'Token Accuracy: {test_correct}/{test_total} = {test_correct / test_total:.15f}')
        print()
        print_entity_statistics(test_entity_correct_counts, test_entity_total_counts)
        print()
        print(test_report)
        save_result('test', tokens, preds, trues)
        return

    params = [
        {'params': model.encoder_t.parameters(), 'lr': args.lr},
        {'params': model.head.parameters(), 'lr': args.lr * 100},
    ]
    if args.encoder_v:
        params.append({'params': model.encoder_v.parameters(), 'lr': args.lr})
        params.append({'params': model.proj.parameters(), 'lr': args.lr * 100})
        params.append({'params': model.txt_proj.parameters(), 'lr': args.lr * 100})
        params.append({'params': model.img_proj.parameters(), 'lr': args.lr * 100})
        params.append({'params': model.itm_head.parameters(), 'lr': args.lr * 100})
        params.append({'params': [model.logit_scale], 'lr': args.lr * 100})
    if args.rnn:
        params.append({'params': model.rnn.parameters(), 'lr': args.lr * 100})
    if args.crf:
        params.append({'params': model.crf.parameters(), 'lr': args.lr * 100})
    if args.gate:
        params.append({'params': model.aux_head.parameters(), 'lr': args.lr * 100})

    optimizer = getattr(torch.optim, args.optim)(params)

    print(args)
    dev_f1s, test_f1s = [], []
    ner_losses, itr_losses = [], []
    best_dev_f1, best_test_f1, best_test_report = 0, 0, None
    best_total = best_correct = best_wrong = 0
    best_entity_correct = {}
    best_entity_total = {}
    for epoch in range(1, args.num_epochs + 1):
        if args.aux:
            itr_loss = train(itr_train_loader, model, optimizer, task='itr', weight=0.05)
            itr_losses.append(itr_loss)
            print(f'loss of image-text relation classification at epoch#{epoch}: {itr_loss:.2f}')

        ner_loss = train(ner_train_loader, model, optimizer, task='ner')
        ner_losses.append(ner_loss)
        print(f'loss of multimodal named entity recognition at epoch#{epoch}: {ner_loss:.2f}')

        dev_f1, dev_report, _, _, _, _, _ = evaluate(model, ner_dev_loader)
        dev_f1s.append(dev_f1)
        (
            test_f1,
            test_report,
            test_total,
            test_correct,
            test_wrong,
            test_entity_correct_counts,
            test_entity_total_counts,
        ) = evaluate(model, ner_test_loader)
        test_f1s.append(test_f1)
        print(f'f1 score on dev set: {dev_f1:.4f}, f1 score on test set: {test_f1:.4f}')
        print(f'测试集共 {test_total} 个 token，预测正确 {test_correct} 个，预测错误 {test_wrong} 个')
        print(f'Token Accuracy: {test_correct}/{test_total} = {test_correct / test_total:.15f}')
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_test_report = test_report
            best_total = test_total
            best_correct = test_correct
            best_wrong = test_wrong
            best_entity_correct = dict(test_entity_correct_counts)
            best_entity_total = dict(test_entity_total_counts)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_model.pt'))

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'epoch{epoch}.pt'))

    print()
    print(f'f1 score on dev set: {best_dev_f1:.4f}, f1 score on test set: {best_test_f1:.4f}')
    print(f'测试集共 {best_total} 个 token，预测正确 {best_correct} 个，预测错误 {best_wrong} 个')
    if best_total > 0:
        print(f'Token Accuracy: {best_correct}/{best_total} = {best_correct / best_total:.15f}')
    else:
        print('Token Accuracy: 无测试样本')
    print()
    print_entity_statistics(best_entity_correct, best_entity_total)
    print()
    print(best_test_report)

    _, _, _, _, _, _, _, train_tokens, train_preds, train_trues = evaluate(model, ner_train_loader, return_preds=True)
    save_result('train', train_tokens, train_preds, train_trues)

    results = {
        'config': vars(args),
        'dev_f1s': dev_f1s,
        'test_f1s': test_f1s,
        'ner_losses': ner_losses,
        'itr_losses': itr_losses,
    }
    file_name = f'log/{args.dataset}/bs{args.bs}_lr{args.lr}_seed{args.seed}.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

# === Windows 多进程入口保护 ===
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
