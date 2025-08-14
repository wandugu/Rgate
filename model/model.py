from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
import torchvision.transforms as T
from PIL import Image
import numpy as np
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.data import Token as FlairToken
from flair.data import Sentence as FlairSentence
from torchcrf import CRF
from data.dataset import MyDataPoint, MyPair
import constants
from model.fine_grained_gate import FineGrainedGate


# constants for model
CLS_POS = 0
SUBTOKEN_PREFIX = '##'
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.feat is None:
            return False
    return True


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x


class MyModel(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            encoder_t: PreTrainedModel,
            hid_dim_t: int,
            encoder_v: nn.Module = None,
            hid_dim_v: int = None,
            token_embedding: TokenEmbeddings = None,
            rnn: bool = None,
            crf: bool = None,
            gate: bool = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = token_embedding
        self.proj = nn.Linear(hid_dim_v, hid_dim_t) if encoder_v else None
        self.aux_head = nn.Linear(hid_dim_t, 2)
        self.fine_gate = FineGrainedGate(hid_dim_t) if (encoder_v and gate) else None
        if self.token_embedding:
            self.hid_dim_t += self.token_embedding.embedding_length
        if rnn:
            hid_dim_rnn = 256
            num_layers = 2
            num_directions = 2
            self.rnn = nn.LSTM(self.hid_dim_t, hid_dim_rnn, num_layers, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hid_dim_rnn * num_directions, constants.LABEL_SET_SIZE)
        else:
            self.rnn = None
            self.head = nn.Linear(self.hid_dim_t, constants.LABEL_SET_SIZE)
        self.crf = CRF(constants.LABEL_SET_SIZE, batch_first=True) if crf else None
        self.gate = gate
        # 新增：视觉输入的标准预处理（与 ImageNet 预训练一致）
        if self.encoder_v is not None:
            self.image_preprocess = T.Compose([
                T.Lambda(lambda im: im.convert("RGB") if hasattr(im, "convert") else im),
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.to(device)

    @classmethod
    def from_pretrained(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = 'model'

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path, use_fast=True)
        encoder_t = AutoModel.from_pretrained(encoder_t_path)
        config = AutoConfig.from_pretrained(encoder_t_path)
        hid_dim_t = config.hidden_size

        if args.encoder_v:
            encoder_v = getattr(torchvision.models, args.encoder_v)()
            encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth'))
            hid_dim_v = encoder_v.fc.in_features
        else:
            encoder_v = None
            hid_dim_v = None

        if args.stacked:
            flair.cache_root = 'model'
            flair.device = device
            token_embedding = StackedEmbeddings([
                WordEmbeddings('crawl'),
                WordEmbeddings('twitter'),
                FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')
            ])
        else:
            token_embedding = None

        return cls(
            device=device,
            tokenizer=tokenizer,
            encoder_t=encoder_t,
            hid_dim_t=hid_dim_t,
            encoder_v=encoder_v,
            hid_dim_v=hid_dim_v,
            token_embedding=token_embedding,
            rnn=args.rnn,
            crf=args.crf,
            gate=args.gate,
        )

    def _bert_forward_with_image(self, inputs, pairs):
        images = [pair.image for pair in pairs]
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)

        # 新增：统一图像为像素张量并编码为视觉序列
        pixels = self._stack_image_batch(images)                 # [B,3,224,224]
        visual_embeds = resnet_encode(self.encoder_v, pixels)    # [B, S, hid_dim_v]
        visual_embeds = self.proj(visual_embeds)
        inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)

        batch_size = visual_embeds.size()[0]
        visual_length = visual_embeds.size()[1]

        attention_mask = inputs.attention_mask
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = inputs.token_type_ids
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

    def _stack_image_batch(self, images):
        """
        将一批图像统一为 [B,3,224,224] 的 float32 Tensor（已 ImageNet 规范化）。
        支持输入为：
          - 自定义对象（如 MyImage），其内部通过 .data 暴露真实图像
          - PIL.Image.Image
          - torch.Tensor（CHW/HWC 均可）
          - numpy.ndarray（HWC）
        """
        def _norm_imagenet_chw(t: torch.Tensor) -> torch.Tensor:
            # t: [3,H,W] float in [0,1]
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
            return (t - mean) / std

        tensors = []
        for img in images:
            base = img.data if hasattr(img, "data") else img  # 解包 MyImage 等自定义类型

            # 分类型处理
            if isinstance(base, Image.Image) or hasattr(base, "convert"):
                # PIL 系：直接走预处理（包含 RGB、Resize、ToTensor、Normalize）
                pil = base.convert("RGB") if hasattr(base, "convert") else base
                t = self.image_preprocess(pil)  # [3,224,224] float 已归一化

            elif isinstance(base, np.ndarray):
                # numpy：假定 HWC，转 PIL 再走预处理
                if base.ndim == 3 and base.shape[-1] in (1, 3):
                    pil = Image.fromarray(base)
                else:
                    raise TypeError(f"Unsupported ndarray shape: {base.shape}")
                t = self.image_preprocess(pil)  # [3,224,224]

            elif isinstance(base, torch.Tensor):
                # Tensor：统一成 CHW，归一化到 [0,1]，再插值到 224，最后做 ImageNet Normalize
                t = base
                # 去 batch 维
                if t.ndim == 4 and t.size(0) == 1:
                    t = t.squeeze(0)
                # HWC -> CHW
                if t.ndim == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1)
                if t.ndim != 3 or t.shape[0] not in (1, 3):
                    raise TypeError(f"Unsupported tensor shape: {tuple(t.shape)}")

                t = t.float()
                # 归一化到 [0,1]
                if t.max() > 1.0:
                    t = t / 255.0
                # 插值到目标尺寸
                if t.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                    t = F.interpolate(
                        t.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
                # ImageNet 归一化
                t = _norm_imagenet_chw(t)

            else:
                # 兜底：尝试常见的自定义持有
                if hasattr(base, "to_pil"):
                    t = self.image_preprocess(base.to_pil())
                elif hasattr(base, "image") and hasattr(base.image, "convert"):
                    t = self.image_preprocess(base.image)
                else:
                    raise TypeError(f"Unsupported image type: {type(base)}")

            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)  # [B,3,224,224]
        return batch




    def ner_encode(self, pairs: List[MyPair]):
        sentence_batch = [pair.sentence for pair in pairs]
        tokens_batch = [[token.text for token in sentence] for sentence in sentence_batch]

        inputs = self.tokenizer(
            tokens_batch,
            is_split_into_words=True,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True
        ).to(self.device)

        if self.encoder_v and self.gate and self.fine_gate is not None:
            text_outputs = self.encoder_t(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_type_ids=inputs.token_type_ids,
                return_dict=True
            )
            feat_batch = text_outputs.last_hidden_state
            images = [pair.image for pair in pairs]
            pixels = self._stack_image_batch(images)                 # [B,3,224,224]
            visual_embeds = resnet_encode(self.encoder_v, pixels)    # [B, S, hid_dim_v]
            visual_embeds = self.proj(visual_embeds)                 # [B, S, hid_dim_t]

            feat_batch, _ = self.fine_gate(feat_batch, visual_embeds)
        elif self.encoder_v:
            outputs = self._bert_forward_with_image(inputs, pairs)
            # 用真实文本长度切回文本特征，避免依赖固定 VISUAL_LENGTH
            feat_batch = outputs.last_hidden_state[:, :inputs.input_ids.size(1)]
        else:
            outputs = self.encoder_t(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_type_ids=inputs.token_type_ids,
                return_dict=True
            )
            feat_batch = outputs.last_hidden_state

        # ⚠ 关键改动：使用 word_ids 精准对齐原词和子词特征
        word_ids_batch = [inputs.word_ids(batch_index=i) for i in range(len(sentence_batch))]

        for sent_idx, (sentence, word_ids, feats) in enumerate(zip(sentence_batch, word_ids_batch, feat_batch)):
            token_feats = [[] for _ in range(len(sentence))]

            for i, word_id in enumerate(word_ids):
                if word_id is None or word_id >= len(sentence):
                    continue
                token_feats[word_id].append(feats[i])

            for i, token in enumerate(sentence):
                if len(token_feats[i]) == 0:
                    token.feat = torch.zeros(self.hid_dim_t, device=self.device)
                else:
                    token.feat = torch.mean(torch.stack(token_feats[i]), dim=0)

            if self.token_embedding is not None:
                flair_sentence = FlairSentence(" ".join([t.text for t in sentence]))
                flair_sentence.tokens = [FlairToken(token.text) for token in sentence]
                self.token_embedding.embed(flair_sentence)
                for token, flair_token in zip(sentence, flair_sentence):
                    token.feat = torch.cat((token.feat, flair_token.embedding))

    def ner_forward(self, pairs: List[MyPair]):
        self.ner_encode(pairs)

        sentences = [pair.sentence for pair in pairs]
        batch_size = len(sentences)
        lengths = [len(sentence) for sentence in sentences]
        max_length = max(lengths)

        feat_list = []
        zero_tensor = torch.zeros(max_length * self.hid_dim_t, device=self.device)
        for sentence in sentences:
            feat_list += [token.feat for token in sentence]
            num_padding = max_length - len(sentence)
            if num_padding > 0:
                padding = zero_tensor[:self.hid_dim_t * num_padding]
                feat_list.append(padding)
        feats = torch.cat(feat_list).view(batch_size, max_length, self.hid_dim_t)

        if self.rnn is not None:
            feats = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
            feats, _ = self.rnn(feats)
            feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)

        logits_batch = self.head(feats)

        labels_batch = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        for i, sentence in enumerate(sentences):
            labels = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            labels_batch[i, :lengths[i]] = labels

        if self.crf:
            mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1
            loss = -self.crf(logits_batch, labels_batch, mask, reduction='mean')
            pred_ids = self.crf.decode(logits_batch, mask)
            pred = [[constants.ID_TO_LABEL[i] for i in ids] for ids in pred_ids]
        else:
            loss = torch.zeros(1, device=self.device)
            for logits, labels, length in zip(logits_batch, labels_batch, lengths):
                loss += F.cross_entropy(logits[:length], labels[:length], reduction='sum')
            loss /= batch_size
            pred_ids = torch.argmax(logits_batch, dim=2).tolist()
            pred = [[constants.ID_TO_LABEL[i] for i in ids[:length]] for ids, length in zip(pred_ids, lengths)]

        return loss, pred

    def itr_forward(self, pairs: List[MyPair]):
        text_batch = [pair.sentence.text for pair in pairs]
        inputs = self.tokenizer(text_batch, padding=True, return_tensors='pt').to(self.device)
        outputs = self._bert_forward_with_image(inputs, pairs)
        feats = outputs.last_hidden_state[:, CLS_POS]
        logits = self.aux_head(feats)

        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        pred = torch.argmax(logits, dim=1).tolist()

        return loss, pred

