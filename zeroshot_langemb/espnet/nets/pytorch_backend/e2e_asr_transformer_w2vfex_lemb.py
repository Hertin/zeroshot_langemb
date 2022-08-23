# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""
import os
from argparse import Namespace
import logging
import math
import json
from copy import deepcopy
import numpy
import numpy as np
from nodevectors import Node2Vec
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
import fairseq
from torch.nn.parameter import Parameter
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import EncoderLang as Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

from espnet.nets.pytorch_backend.graph_convolutional_nets.lang_gcn import LangGCN

from espnet.utils.cli_utils import strtobool

class W2VEncoder(torch.nn.Module):
    def __init__(self, wav2vec, idim, adim, args):
        torch.nn.Module.__init__(self)
        self.wav2vec = wav2vec
        self.w2v_linear = nn.Linear(idim, adim)
        self.args = args

    def forward(self, audio_pad, audio_pad_mask):
        hs_pad, pad_mask = self.wav2vec.extract_features(audio_pad, audio_pad_mask)
        hs_pad = self.w2v_linear(hs_pad)
        hs_mask = ~pad_mask
        return hs_pad, hs_mask


class LangEmb(nn.Module):
    def __init__(self, args):
        super(LangEmb, self).__init__()
        self.args = args

        with open(args.glotto_all_phonemes, 'r') as f:
            self.glotto_all_phonemes = sorted(json.load(f))
        with open(args.lang2glottoph, 'r') as f:
            self.lang2ph = json.load(f)
        with open(args.lang2glotto, 'r') as f:
            self.lang2glotto = json.load(f)

        
        self.all_langs = sorted([l for l in self.lang2ph.keys()])
        logging.warning(f'LangEmb All langs {self.all_langs}')

        if args.lang_emb_type == 'all':
            logging.warning('Use all features')
            self.input_dim = args.lgcn_n2v_dim + len(self.glotto_all_phonemes)
        elif args.lang_emb_type =='phoible':
            logging.warning('Use phoible features')
            self.input_dim = len(self.glotto_all_phonemes)
        elif args.lang_emb_type =='glotto':
            logging.warning('Use glotto features')
            self.input_dim = args.lgcn_n2v_dim
        else:
            raise ValueError(f'lang_emb_type {args.lang_emb_type} not implemented')
        self.hidden_dim = args.lgcn_hidden_dim
        self.output_dim = args.lgcn_output_dim
        self.num_layer = args.lgcn_num_layer
        self.dropout = args.dropout_rate

        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(dim_list[tt], dim_list[tt+1]) for tt in range(self.num_layer)
        ])
        self.final_linear = nn.Linear(dim_list[-2], dim_list[-1])
        
        dtype = 'half' if args.train_dtype in ("O0", "O1", "O2", "O3") else 'float'
        device = torch.device("cuda" if args.ngpu > 0 else "cpu")

        # 1. g2v embedding
        if os.path.exists(f'{args.lgcn_g2v_path}.npy'):
            logging.warning('load g2v npy directly')
            g = nx.read_gpickle(args.lgcn_graph_path)
            node2idx = {l: i for i, l in enumerate(g.nodes)}
            
            n2v_embedding = np.load(f'{args.lgcn_g2v_path}.npy')
            lang_indices = [node2idx[self.lang2glotto[l]] for l in self.all_langs]
            logging.warning(f'lang_indices  {lang_indices}')
            self.n2v_embedding = n2v_embedding[lang_indices]
            logging.warning(f'n2v_embedding  {self.n2v_embedding.shape}')
        else:
            self.g2v = Node2Vec.load(args.lgcn_g2v_path)
            self.n2v_embedding = np.array([self.g2v.predict(l) for l in self.g.nodes])
            self.g2v = Node2Vec.load(args.lgcn_g2v_path)
            self.n2v_embedding = np.array([self.g2v.predict(self.lang2glotto[l]) for l in self.all_langs])
        
        # 2. one hot for phoneme used
        
        self.ph_embedding = np.zeros((len(self.lang2ph), len(self.glotto_all_phonemes)))
        for lang, phones in self.lang2ph.items():
            for ph in phones:
                self.ph_embedding[self.all_langs.index(lang), self.glotto_all_phonemes.index(ph)] = 1

        # 3. concate embeddings
        if args.lang_emb_type == 'all':
            embedding = np.concatenate([self.n2v_embedding, self.ph_embedding], axis=1)
        elif args.lang_emb_type == 'phoible':
            embedding = self.ph_embedding
        elif args.lang_emb_type == 'glotto':
            embedding = self.n2v_embedding
        
        self.embedding = Parameter(torch.from_numpy(embedding.astype(float)), requires_grad=False)
        logging.warning(f'lang embedding size {self.embedding.size()}')

    def forward(self, langs):
        lang_ids = [self.all_langs.index(lang) for lang in langs]
        state = self.embedding[lang_ids]
        # logging.warning(f'lemb forward 1 {state.size()}')
        for tt in range(self.num_layer):
            state = self.linear_layers[tt](state)
            state = F.relu(state)
            state = F.dropout(state, self.dropout, training=self.training)
        state = self.final_linear(state) # B x F
        # logging.warning(f'lemb forward 2 {langs} {lang_ids} {state.size()}')
        return state

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        group.add_argument(
            "--lgcn-n2v-dim",
            default=128,
            type=int,
            help="lgcn-n2v-dim",
        )

        group.add_argument(
            "--lgcn-g2v-path",
            default=None,
            type=str,
            help="lgcn-n2v-dim",
        )

        group.add_argument(
            "--lgcn-graph-path",
            default=None,
            type=str,
            help="lgcn-n2v-dim",
        )

        group.add_argument(
            "--lgcn-lang2lid-path",
            default=None,
            type=str,
            help="lgcn-n2v-dim",
        )

        group.add_argument(
            "--glotto-all-phonemes",
            default=None,
            type=str,
            help="glotto-all-phonemes",
        )

        group.add_argument(
            "--lang2glottoph",
            default=None,
            type=str,
            help="lang2glottoph",
        )

        group.add_argument(
            "--lang2glotto",
            default=None,
            type=str,
            help="lang2glotto",
        )

        group.add_argument(
            "--lgcn-hidden-dim",
            default=[128],
            type=lambda s: [int(n) for n in s.split(",")],
            help="lgcn-hidden-dim",
        )

        group.add_argument(
            "--lgcn-output-dim",
            default=128,
            type=int,
            help="lgcn-output-dim",
        )
        
        group.add_argument(
            "--lgcn-num-layer",
            default=3,
            type=int,
            help="lgcn-output-dim",
        )

        group.add_argument(
            "--wav2vec-fix-extractor",
            default=True,
            type=strtobool,
            help="wav2vec-fix-extractor",
        )

        group.add_argument(
            "--lang-emb-type",
            default='all',
            type=str,
            help="lang2glotto",
        )

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)
        self.args = args
        self.idim = idim if not args.wav2vec_feature else args.wav2vec_idim

        logging.warning(f'idim {self.idim}')
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        
        wav2vec, wav2vec_cfg = fairseq.checkpoint_utils.load_model_ensemble([args.wav2vec_path])
        
        wav2vec = wav2vec[0]
        self.wav2vec_cfg = deepcopy(wav2vec.cfg)
        self.feature_extractor = deepcopy(wav2vec.feature_extractor)
        if args.wav2vec_fix_extractor:
            logging.warning(f'fix feature extractor')
            for parameter in self.feature_extractor.parameters():
                # logging.warning(f'parameter of feature extractor are fixed {parameter.name}')
                parameter.requires_grad = False

        del wav2vec

        self.lemb = LangEmb(args)

        # self.encoder = W2VEncoder(self.wav2vec, self.idim, args.adim, args)
        self.encoder = Encoder(
            idim=self.idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            langemb_dim=args.lgcn_output_dim
        )
        

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            self.criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha

        self.lang2phid = None
        if args.lang2ph is not None:
            with open(args.lang2ph, 'r') as f:
                self.lang2ph = json.load(f)

            self.lang2phid = {}
            for lang, phones in self.lang2ph.items(): 
                phoneset = set(phones + ['<blank>', '<unk>', '<space>', '<eos>'])
                phoneset = phoneset.intersection(self.args.char_list)
                self.lang2phid[lang] = list(map(self.args.char_list.index, phoneset))

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, 
                ctc_type=args.ctc_type, reduce=False,
                length_average=args.warpctc_length_average,
                lang2phid=self.lang2phid
            )
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size).float() / stride + 1)

        conv_cfg_list = eval(self.wav2vec_cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])

        return input_lengths.to(torch.long)

    def forward(self, audio_pad, audio_lens, ys_pad, lang_labels):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # audio_pad_mask = make_pad_mask(audio_lens)
        # audio_pad_mask
        # hs_pad, hs_mask 
        # logging.warning(f'audio_pad size {audio_pad.size()}')
        # logging.warning(f'lang labels {lang_labels} {lang_label_indices}')
        # lang_labels = np.array(lang_labels)[lang_label_indices.cpu().numpy()]

        langembs = self.lemb(lang_labels)

        # logging.warning(f'{ lang_labels} lgcn out size {langembs.size()} {langembs.type()}')

        features = self.feature_extractor(audio_pad)
        features = features.transpose(1, 2)
        # features = self.layer_norm(features)
        # logging.warning(f'features size {features.size()}')
        output_lengths = self._get_feat_extract_output_lengths(audio_lens)
        # logging.warning(f'features output_lengths {output_lengths}')
        src_mask = make_non_pad_mask(output_lengths.tolist()).to(audio_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(langembs, features, src_mask)

        batch_size = hs_pad.size(0)
        ys = [y[y != self.ignore_id] for y in ys_pad]
        olens = torch.from_numpy(numpy.fromiter((x.size(0) for x in ys), dtype=numpy.int32))

        # 1. forward encoder
        # remove utterances that are shorter than target
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        valid_indices = hs_len.cpu().int() > olens
        # hs_pad, hs_mask, hs_len, ys_pad = hs_pad[valid_indices], hs_mask[valid_indices], hs_len[valid_indices], ys_pad[valid_indices]
        invalid = False
        if torch.sum(valid_indices) < batch_size:
            # logging.warning(f'Remove {batch_size - torch.sum(valid_indices)} invalid utterances')
            invalid = True
        # batch_size = hs_pad.size(0) # update new batch_size

        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:            
            # loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            loss_ctc_nonreduce = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            invalid_idx = torch.isinf(loss_ctc_nonreduce) | torch.isnan(loss_ctc_nonreduce)
            if torch.sum(invalid_idx != 0):
                logging.warning(f'Invalid ctc loss {invalid} num invalid {torch.sum(invalid_idx != 0)} {loss_ctc_nonreduce[invalid_idx]}')

            loss_ctc_nonreduce[invalid_idx] = 0
            # loss_ctc_nonreduce[torch.isnan(loss_ctc_nonreduce)] = 0
            loss_ctc = loss_ctc_nonreduce[~invalid_idx].mean() if any(~invalid_idx) else torch.FloatTensor([0]).to(loss_ctc_nonreduce.device)

            

            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # if invalid:
        #     logging.warning(f'ctc loss {loss_ctc}')
        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        # logging.warning(f'{self.loss}')
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, audio_pad, **kwargs):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        lang_labels = kwargs.get('lang_labels')
        langembs = self.lemb(lang_labels)

        audio_pad = torch.as_tensor(audio_pad).unsqueeze(0)
        features = self.feature_extractor(audio_pad)
        features = features.transpose(1, 2)
        if kwargs.get('lang_agnostic', False):
            _, emb_size = langembs.size()
            hs_pad, hs_mask = self.encoder(torch.zeros(langembs.size()).to(langembs.device), features, None)
        else:
            hs_pad, hs_mask = self.encoder(langembs, features, None)

        return hs_pad.squeeze(0)

    def encode_with_length(self, audio_pad, audio_lens, lang_labels=None, mask_phoneme=False, **kwargs):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        
        self.eval()

        langembs = self.lemb(lang_labels)

        features = self.feature_extractor(audio_pad)
        features = features.transpose(1, 2)

        output_lengths = self._get_feat_extract_output_lengths(audio_lens)
        # logging.warning(f'features output_lengths {output_lengths}')
        src_mask = make_non_pad_mask(output_lengths.tolist()).to(audio_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(langembs, features, src_mask)
        self.hs_pad = hs_pad

        batch_size = audio_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)

        
        lang_labels_for_masking = kwargs['lang_labels_for_masking'] if mask_phoneme else None
        # logging.warning(f'{mask_phoneme} {lang_labels_for_masking} {lang_labels}')
        # raise
        log_probs = self.ctc.log_softmax(hs_pad.view(batch_size, -1, self.adim), lang_labels=lang_labels_for_masking)
        # , lang_labels=lang_labels
        return log_probs, hs_len

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, lang_labels):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, lang_labels)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, lang_labels):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, lang_labels)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
