from __future__ import annotations

from typing import List, Optional, Tuple
import copy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

try:
    from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
except ImportError:
    PreTrainedTokenizer = PreTrainedModel = AutoTokenizer = AutoModel = AutoConfig = None

try:
    import flair
    from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
    from flair.data import Token as FlairToken
    from flair.data import Sentence as FlairSentence
except ImportError:  # Flair is only needed when --stacked is enabled.
    flair = None
    WordEmbeddings = FlairEmbeddings = StackedEmbeddings = TokenEmbeddings = None
    FlairToken = FlairSentence = None

try:
    from torchcrf import CRF
except ImportError:
    try:
        from TorchCRF import CRF
    except ImportError:
        CRF = None

from data.dataset import MyDataPoint, MyPair
import constants


CLS_POS = 0
PREVIEW_IMAGE_SIZE = 112
EPS = 1e-6


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


class LightweightCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            n_heads = 1
        self.text_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, text_states: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        query = self.text_norm(text_states)
        key_value = self.visual_norm(visual_tokens)
        context, _ = self.attn(query=query, key=key_value, value=key_value, need_weights=False)
        fused = text_states + self.dropout(context)
        return fused + self.dropout(self.ffn(fused))


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
            rgate_stage: int = 3,
            theta0: float = 0.5,
            theta1: float = 0.5,
            use_delta: float = 0.0,
            budget_tau: float = 0.4,
            alpha_itm: float = 1.0,
            beta_nce: float = 0.5,
            mu_use: float = 1.0,
            lambda_budget: float = 0.2,
            lambda_cost: float = 0.05,
            fuse_cost: float = 1.0,
            eta_rl: float = 1.0,
            beta_kl: float = 0.01,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        if crf and CRF is None:
            raise ImportError("CRF support requires torchcrf or TorchCRF to be installed.")

        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.base_hid_dim_t = hid_dim_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = token_embedding
        self.gate = bool(gate)

        self.proj = nn.Linear(hid_dim_v, hid_dim_t) if encoder_v else None
        self.fusion = LightweightCrossAttention(hid_dim_t) if encoder_v else None
        self.aux_head = nn.Linear(hid_dim_t, 2)

        if encoder_v and self.gate:
            self.preview_encoder = torchvision.models.resnet18()
            preview_dim = self.preview_encoder.fc.in_features
            self.preview_proj = nn.Linear(preview_dim, hid_dim_t)
            self.pair_proj = nn.Sequential(
                nn.Linear(hid_dim_t * 4, hid_dim_t),
                nn.LayerNorm(hid_dim_t),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hid_dim_t, hid_dim_t),
                nn.GELU(),
            )
            self.early_gate = nn.Sequential(
                nn.Linear(hid_dim_t * 2 + 1, hid_dim_t // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hid_dim_t // 2, 1),
            )
            self.relation_gate = nn.Sequential(
                nn.Linear(hid_dim_t + 1, hid_dim_t // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hid_dim_t // 2, 1),
            )
            self.relation_gate_ema = copy.deepcopy(self.relation_gate)
            for parameter in self.relation_gate_ema.parameters():
                parameter.requires_grad_(False)
            self.itm_head = nn.Linear(hid_dim_t, 1)
            self.nce_head = nn.Linear(hid_dim_t, 1)
            self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        else:
            self.preview_encoder = None
            self.preview_proj = None
            self.pair_proj = None
            self.early_gate = None
            self.relation_gate = None
            self.relation_gate_ema = None
            self.itm_head = None
            self.nce_head = None
            self.logit_scale = None

        self.rgate_stage = int(rgate_stage)
        self.theta0 = theta0
        self.theta1 = theta1
        self.use_delta = use_delta
        self.budget_tau = budget_tau
        self.alpha_itm = alpha_itm
        self.beta_nce = beta_nce
        self.mu_use = mu_use
        self.lambda_budget = lambda_budget
        self.lambda_cost = lambda_cost
        self.fuse_cost = fuse_cost
        self.eta_rl = eta_rl
        self.beta_kl = beta_kl
        self.ema_decay = ema_decay
        self.register_buffer("rl_reward_baseline", torch.tensor(0.0))
        self.last_gate_stats = {}

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
        self.to(device)

    @classmethod
    def from_pretrained(cls, args):
        if AutoTokenizer is None or AutoModel is None or AutoConfig is None:
            raise ImportError("Loading pretrained text encoders requires transformers to be installed.")
        device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        models_path = 'model'

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)
        encoder_t = AutoModel.from_pretrained(encoder_t_path)
        config = AutoConfig.from_pretrained(encoder_t_path)
        hid_dim_t = config.hidden_size

        if args.encoder_v:
            encoder_v = getattr(torchvision.models, args.encoder_v)()
            visual_weight_path = Path(models_path) / 'cnn' / f'{args.encoder_v}.pth'
            if not visual_weight_path.exists():
                raise FileNotFoundError(
                    f'Visual encoder weights not found: {visual_weight_path}. '
                    f'Place {args.encoder_v}.pth there or pass --encoder_v resnet101 to use ResNet-101.'
                )
            encoder_v.load_state_dict(torch.load(str(visual_weight_path), map_location='cpu'))
            hid_dim_v = encoder_v.fc.in_features
        else:
            encoder_v = None
            hid_dim_v = None

        if args.stacked:
            if flair is None:
                raise ImportError("The --stacked option requires flair to be installed.")
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
            rgate_stage=getattr(args, 'rgate_stage', 3),
            theta0=getattr(args, 'theta0', 0.5),
            theta1=getattr(args, 'theta1', 0.5),
            use_delta=getattr(args, 'use_delta', 0.0),
            budget_tau=getattr(args, 'budget_tau', 0.4),
            alpha_itm=getattr(args, 'alpha_itm', 1.0),
            beta_nce=getattr(args, 'beta_nce', 0.5),
            mu_use=getattr(args, 'mu_use', 1.0),
            lambda_budget=getattr(args, 'lambda_budget', 0.2),
            lambda_cost=getattr(args, 'lambda_cost', 0.05),
            fuse_cost=getattr(args, 'fuse_cost', 1.0),
            eta_rl=getattr(args, 'eta_rl', 1.0),
            beta_kl=getattr(args, 'beta_kl', 0.01),
            ema_decay=getattr(args, 'ema_decay', 0.99),
        )

    def set_rgate_stage(self, stage: int):
        self.rgate_stage = int(stage)

    def update_ema(self):
        if self.relation_gate is None or self.relation_gate_ema is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.relation_gate_ema.parameters(), self.relation_gate.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def _token_type_ids(self, inputs):
        token_type_ids = getattr(inputs, 'token_type_ids', None)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(inputs.input_ids)
        return token_type_ids

    def _tokenize_sentences(self, sentences):
        tokens_batch = [[token.text for token in sentence] for sentence in sentences]
        return self.tokenizer(
            tokens_batch,
            is_split_into_words=True,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True
        ).to(self.device)

    def _encode_text(self, inputs) -> torch.Tensor:
        outputs = self.encoder_t(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            token_type_ids=self._token_type_ids(inputs),
            return_dict=True
        )
        return outputs.last_hidden_state

    def _encode_visual_tokens(self, images: List[MyDataPoint]) -> torch.Tensor:
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)
        if not use_cache(self.encoder_v, images):
            visual_embeds = resnet_encode(self.encoder_v, visual_embeds)
        return self.proj(visual_embeds)

    def _encode_preview(self, images: List[MyDataPoint]) -> torch.Tensor:
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)
        visual_embeds = F.interpolate(
            visual_embeds,
            size=(PREVIEW_IMAGE_SIZE, PREVIEW_IMAGE_SIZE),
            mode='bilinear',
            align_corners=False,
        )
        preview_tokens = resnet_encode(self.preview_encoder, visual_embeds)
        return self.preview_proj(preview_tokens.mean(dim=1))

    def _bert_forward_with_image(self, inputs, pairs):
        images = [pair.image for pair in pairs]
        visual_embeds = self._encode_visual_tokens(images)
        return self._bert_forward_with_visual(inputs, visual_embeds)

    def _bert_forward_with_visual(self, inputs, visual_embeds):
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)
        inputs_embeds = torch.cat((textual_embeds, visual_embeds), dim=1)

        batch_size = visual_embeds.size(0)
        visual_length = visual_embeds.size(1)

        attention_mask = inputs.attention_mask
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = self._token_type_ids(inputs)
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

    def _word_features_from_subwords(self, sentences, inputs, subword_states: torch.Tensor) -> torch.Tensor:
        batch_size = len(sentences)
        max_length = max(len(sentence) for sentence in sentences)
        word_feats = torch.zeros(batch_size, max_length, subword_states.size(-1), device=self.device)
        word_ids_batch = [inputs.word_ids(batch_index=i) for i in range(batch_size)]

        for batch_idx, (sentence, word_ids) in enumerate(zip(sentences, word_ids_batch)):
            seen = set()
            for pos, word_id in enumerate(word_ids):
                if word_id is None or word_id >= len(sentence) or word_id in seen:
                    continue
                word_feats[batch_idx, word_id] = subword_states[batch_idx, pos]
                seen.add(word_id)
        return word_feats

    def _label_tensors(self, sentences, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        batch_size = len(sentences)
        labels = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
        lengths = []
        for idx, sentence in enumerate(sentences):
            length = len(sentence)
            lengths.append(length)
            labels[idx, :length] = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            mask[idx, :length] = True
        return labels, mask, lengths

    def _token_embedding_batch(self, sentences, max_length: int) -> Optional[torch.Tensor]:
        if self.token_embedding is None:
            return None
        extra = torch.zeros(
            len(sentences),
            max_length,
            self.token_embedding.embedding_length,
            device=self.device,
        )
        for batch_idx, sentence in enumerate(sentences):
            flair_sentence = FlairSentence(" ".join([token.text for token in sentence]))
            flair_sentence.tokens = [FlairToken(token.text) for token in sentence]
            self.token_embedding.embed(flair_sentence)
            for token_idx, flair_token in enumerate(flair_sentence):
                extra[batch_idx, token_idx] = flair_token.embedding.to(self.device)
        return extra

    def _decoder_logits(self, word_feats: torch.Tensor, lengths: List[int], extra_feats=None) -> torch.Tensor:
        feats = torch.cat((word_feats, extra_feats), dim=-1) if extra_feats is not None else word_feats
        if self.rnn is not None:
            max_length = feats.size(1)
            packed = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
            feats, _ = self.rnn(packed)
            feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True, total_length=max_length)
        return self.head(feats)

    def _loss_from_logits(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            mask: torch.Tensor,
            lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.crf:
            try:
                loss_per_example = -self.crf(logits, labels, mask, reduction='none')
            except TypeError:
                loss_per_example = -self.crf(logits, labels, mask)
                if loss_per_example.dim() == 0:
                    loss_per_example = loss_per_example.repeat(logits.size(0))
            return loss_per_example.mean(), loss_per_example

        token_losses = F.cross_entropy(
            logits.reshape(-1, constants.LABEL_SET_SIZE),
            labels.reshape(-1),
            reduction='none',
        ).view_as(labels)
        loss_per_example = (token_losses * mask.float()).sum(dim=1)
        return loss_per_example.mean(), loss_per_example

    def _pred_from_logits(self, logits: torch.Tensor, mask: torch.Tensor, lengths: List[int]):
        if self.crf:
            pred_ids = self.crf.decode(logits, mask)
            return [[constants.ID_TO_LABEL[idx] for idx in ids] for ids in pred_ids]
        pred_ids = torch.argmax(logits, dim=2).tolist()
        return [[constants.ID_TO_LABEL[idx] for idx in ids[:length]] for ids, length in zip(pred_ids, lengths)]

    def _decode_word_features(self, word_feats, sentences, extra_feats=None, return_pred=True):
        labels, mask, lengths = self._label_tensors(sentences, word_feats.size(1))
        logits = self._decoder_logits(word_feats, lengths, extra_feats=extra_feats)
        loss, loss_per_example = self._loss_from_logits(logits, labels, mask, lengths)
        pred = self._pred_from_logits(logits, mask, lengths) if return_pred else None
        return loss, loss_per_example, pred, logits, mask, lengths

    def _uncertainty_from_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=-1)
        uncertainty = (entropy * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        return uncertainty.unsqueeze(-1).detach()

    def _pair_features(self, c_t: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        pair_input = torch.cat([c_t, c_i, c_t * c_i, (c_t - c_i).abs()], dim=-1)
        return self.pair_proj(pair_input)

    def _pair_feature_matrix(self, c_t: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        batch_size = c_t.size(0)
        c_t_exp = c_t[:, None, :].expand(batch_size, batch_size, -1)
        c_i_exp = c_i[None, :, :].expand(batch_size, batch_size, -1)
        pair_input = torch.cat([c_t_exp, c_i_exp, c_t_exp * c_i_exp, (c_t_exp - c_i_exp).abs()], dim=-1)
        return self.pair_proj(pair_input.reshape(batch_size * batch_size, -1)).view(batch_size, batch_size, -1)

    def _bernoulli_kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp(EPS, 1.0 - EPS)
        q = q.clamp(EPS, 1.0 - EPS)
        return p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()

    def _relation_losses(self, c_t, c_i, neg_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = c_t.size(0)
        zero = torch.tensor(0.0, device=self.device)
        if batch_size < 2:
            return zero, zero

        pair_matrix = self._pair_feature_matrix(c_t, c_i)
        pos_pair = pair_matrix[torch.arange(batch_size, device=self.device), torch.arange(batch_size, device=self.device)]
        neg_pair = pair_matrix[torch.arange(batch_size, device=self.device), neg_idx]

        itm_logits = self.itm_head(torch.cat([pos_pair, neg_pair], dim=0)).squeeze(-1)
        itm_labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0).to(self.device)
        loss_itm = F.binary_cross_entropy_with_logits(itm_logits, itm_labels)

        nce_scores = self.logit_scale.exp() * self.nce_head(pair_matrix).squeeze(-1)
        target = torch.arange(batch_size, device=self.device)
        loss_nce = 0.5 * (F.cross_entropy(nce_scores, target) + F.cross_entropy(nce_scores.t(), target))
        return loss_itm, loss_nce

    def _gate_training_forward(self, pairs, text_states, text_word_feats, text_loss_per_example, text_logits, mask, extra_feats, neg_idx):
        c_t = text_states[:, CLS_POS]
        images = [pair.image for pair in pairs]
        c_i_preview = self._encode_preview(images)
        p0 = torch.sigmoid(self.early_gate(torch.cat([c_t, c_i_preview, self._uncertainty_from_logits(text_logits, mask)], dim=-1))).squeeze(-1)

        visual_tokens = self._encode_visual_tokens(images)
        c_i = visual_tokens.mean(dim=1)
        c_ti = self._pair_features(c_t, c_i)
        uncertainty = self._uncertainty_from_logits(text_logits, mask)
        relation_input = torch.cat([c_ti, uncertainty], dim=-1)
        p1 = torch.sigmoid(self.relation_gate(relation_input)).squeeze(-1)

        fused_word_feats = self.fusion(text_word_feats, p1[:, None, None] * visual_tokens)
        _, mm_loss_per_example, _, _, _, _ = self._decode_word_features(
            fused_word_feats,
            [pair.sentence for pair in pairs],
            extra_feats=extra_feats,
            return_pred=False,
        )

        y_use = ((text_loss_per_example.detach() - mm_loss_per_example.detach()) > self.use_delta).float()
        loss_use = F.binary_cross_entropy(p0.clamp(EPS, 1.0 - EPS), y_use)
        loss_budget = (p0.mean() - self.budget_tau).pow(2)
        loss_itm, loss_nce = self._relation_losses(c_t, c_i, neg_idx)

        loss_rl = torch.tensor(0.0, device=self.device)
        selected_word_feats = fused_word_feats
        if self.rgate_stage >= 3 and self.training:
            sample = torch.bernoulli(p1.detach()).to(self.device)
            selected_word_feats = sample[:, None, None] * fused_word_feats + (1.0 - sample)[:, None, None] * text_word_feats
            _, selected_loss_per_example, _, _, _, _ = self._decode_word_features(
                selected_word_feats,
                [pair.sentence for pair in pairs],
                extra_feats=extra_feats,
                return_pred=False,
            )
            reward = (
                text_loss_per_example.detach()
                - selected_loss_per_example.detach()
                - self.lambda_cost * sample * self.fuse_cost
            )
            with torch.no_grad():
                self.rl_reward_baseline.mul_(0.9).add_(reward.mean(), alpha=0.1)
            log_prob = sample * p1.clamp(EPS, 1.0 - EPS).log() + (1.0 - sample) * (1.0 - p1).clamp(EPS, 1.0).log()
            policy_loss = -self.eta_rl * ((reward - self.rl_reward_baseline).detach() * log_prob).mean()
            with torch.no_grad():
                q = torch.sigmoid(self.relation_gate_ema(relation_input)).squeeze(-1)
            kl_loss = self._bernoulli_kl(p1, q).mean()
            loss_rl = policy_loss + self.beta_kl * kl_loss

        selected_loss, _, pred, _, _, _ = self._decode_word_features(
            selected_word_feats,
            [pair.sentence for pair in pairs],
            extra_feats=extra_feats,
            return_pred=True,
        )

        self.last_gate_stats = {
            "p0": p0.detach(),
            "p1": p1.detach(),
            "cov0": (p0 >= self.theta0).float().mean().detach(),
            "cov1": ((p0 >= self.theta0) & (p1 >= self.theta1)).float().mean().detach(),
            "stage": self.rgate_stage,
        }
        total_loss = (
            selected_loss
            + self.alpha_itm * loss_itm
            + self.beta_nce * loss_nce
            + self.mu_use * loss_use
            + self.lambda_budget * loss_budget
            + loss_rl
        )
        return total_loss, pred

    def _gate_inference_forward(self, pairs, text_states, text_word_feats, text_logits, mask, extra_feats):
        sentences = [pair.sentence for pair in pairs]
        c_t = text_states[:, CLS_POS]
        images = [pair.image for pair in pairs]
        uncertainty = self._uncertainty_from_logits(text_logits, mask)
        c_i_preview = self._encode_preview(images)
        p0 = torch.sigmoid(self.early_gate(torch.cat([c_t, c_i_preview, uncertainty], dim=-1))).squeeze(-1)

        selected_word_feats = text_word_feats.clone()
        p1_all = torch.zeros_like(p0)
        active = p0 >= self.theta0
        inject_all = torch.zeros_like(active)

        if active.any():
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            active_images = [images[idx.item()] for idx in active_idx]
            visual_tokens = self._encode_visual_tokens(active_images)
            c_i = visual_tokens.mean(dim=1)
            c_t_active = c_t.index_select(0, active_idx)
            relation_input = torch.cat([
                self._pair_features(c_t_active, c_i),
                uncertainty.index_select(0, active_idx),
            ], dim=-1)
            p1_active = torch.sigmoid(self.relation_gate(relation_input)).squeeze(-1)
            p1_all[active_idx] = p1_active
            inject = p1_active >= self.theta1
            inject_all[active_idx] = inject
            if inject.any():
                active_text = text_word_feats.index_select(0, active_idx)
                fused_active = self.fusion(active_text, p1_active[:, None, None] * visual_tokens)
                selected_active = torch.where(inject[:, None, None], fused_active, active_text)
                selected_word_feats[active_idx] = selected_active

        loss, _, pred, _, _, _ = self._decode_word_features(
            selected_word_feats,
            sentences,
            extra_feats=extra_feats,
            return_pred=True,
        )
        self.last_gate_stats = {
            "p0": p0.detach(),
            "p1": p1_all.detach(),
            "cov0": active.float().mean().detach(),
            "cov1": inject_all.float().mean().detach(),
            "stage": "inference",
        }
        return loss, pred

    def ner_encode(self, pairs: List[MyPair]):
        sentences = [pair.sentence for pair in pairs]
        inputs = self._tokenize_sentences(sentences)
        text_states = self._encode_text(inputs)
        word_feats = self._word_features_from_subwords(sentences, inputs, text_states)
        extra_feats = self._token_embedding_batch(sentences, word_feats.size(1))
        if extra_feats is not None:
            word_feats = torch.cat((word_feats, extra_feats), dim=-1)
        for sentence, feats in zip(sentences, word_feats):
            for token_idx, token in enumerate(sentence):
                token.feat = feats[token_idx]

    def ner_forward(self, batch):
        pairs = batch["pairs"] if isinstance(batch, dict) else batch
        sentences = [pair.sentence for pair in pairs]
        inputs = self._tokenize_sentences(sentences)
        text_states = self._encode_text(inputs)
        text_word_feats = self._word_features_from_subwords(sentences, inputs, text_states)
        extra_feats = self._token_embedding_batch(sentences, text_word_feats.size(1))

        text_loss, text_loss_per_example, text_pred, text_logits, mask, _ = self._decode_word_features(
            text_word_feats,
            sentences,
            extra_feats=extra_feats,
            return_pred=True,
        )

        if self.encoder_v is None or (self.gate and self.rgate_stage == 1):
            return text_loss, text_pred

        if self.encoder_v is not None and not self.gate:
            visual_tokens = self._encode_visual_tokens([pair.image for pair in pairs])
            fused_word_feats = self.fusion(text_word_feats, visual_tokens)
            loss, _, pred, _, _, _ = self._decode_word_features(
                fused_word_feats,
                sentences,
                extra_feats=extra_feats,
                return_pred=True,
            )
            return loss, pred

        if not self.training:
            return self._gate_inference_forward(pairs, text_states, text_word_feats, text_logits, mask, extra_feats)

        batch_size = len(pairs)
        if isinstance(batch, dict) and "neg_idx" in batch:
            neg_idx = batch["neg_idx"].to(self.device)
        else:
            neg_idx = torch.roll(torch.arange(batch_size, device=self.device), shifts=1)
        if batch_size > 0:
            neg_idx = neg_idx.remainder(batch_size)

        return self._gate_training_forward(
            pairs,
            text_states,
            text_word_feats,
            text_loss_per_example,
            text_logits,
            mask,
            extra_feats,
            neg_idx,
        )

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
