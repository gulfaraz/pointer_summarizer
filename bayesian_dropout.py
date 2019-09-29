# Except for the pytorch part content of this file
# is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import time
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import trange

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config

from training_ptr_gen.model import Model
from training_ptr_gen.train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()


def get_input_from_batch_wrapper(batch, use_cuda):
    a = get_input_from_batch(batch, use_cuda)
    return dict(
        enc_batch=a[0],
        enc_padding_mask=a[1],
        enc_lens=a[2],
        enc_batch_extend_vocab=a[3],
        extra_zeros=a[4],
        c_t_0=a[5],
        coverage_t_0=a[6],
    )


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs

        if not isinstance(state, list):
            state = [state]
        if not isinstance(context, list):
            context = [context]
        if not isinstance(coverage, list):
            coverage = [coverage]

        self.states = state
        self.contexts = context
        self.coverages = coverage

    def iterator(self):
        return zip(
            self.tokens,
            self.log_probs,
            self.states,
            self.contexts,
            self.coverages,
        )

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=self.states + [state],
            context=self.contexts + [context],
            coverage=self.coverages + [coverage],
        )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    @property
    def state(self):
        return self.states[-1]

    @property
    def context(self):
        return self.contexts[-1]

    @property
    def coverage(self):
        return self.coverages[-1]


class StopWatch:
    def __init__(self):
        self.start = time.time()

    def log(self, message):
        time_ = time.time()
        print(message + " {} seconds".format(time_ - self.start))
        self.start = time


class BayesianDropout:
    def __init__(self, model_file_path):

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                          batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        # we use training mode, so dropout is activated
        self.model = Model(model_file_path, is_eval=False)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def _clip_tokens(self, tokens):
        tokens = [t if t < self.vocab.size()
                  else self.vocab.word2id(data.UNKNOWN_TOKEN)
                  for t in tokens]
        return tokens

    def _get_latest_tokens(self, beam):
        latest_tokens = map(lambda b: b.latest_token, beam)
        latest_tokens = self._clip_tokens(latest_tokens)
        return latest_tokens

    def beam_search(self, batch):
        # TODO replace this with allenNLP's beam search?
        # batch should have only one example

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden \
            = self.model.encoder(enc_batch, enc_lens)

        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation,
        # it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = self._get_latest_tokens(beams)

            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()

            all_state_h = [h.state[0] for h in beams]
            all_state_c = [h.state[1] for h in beams]
            all_context = [h.context for h in beams]

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                     torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t \
                = self.model.decoder(
                    y_t_1,
                    s_t_1,
                    encoder_outputs,
                    encoder_feature,
                    enc_padding_mask,
                    c_t_1,
                    extra_zeros,
                    enc_batch_extend_vocab,
                    coverage_t_1,
                    steps
                )

            log_probs = torch.log(final_dist)
            k = config.beam_size * 2
            topk_log_probs, topk_ids = torch.topk(log_probs, k)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                # for each of the top 2*beam_size hyps:
                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(
                        results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    @torch.no_grad()
    def run_bayesian_dropout(
            self, batch, conditioning_summary, num_experiments
    ):
        # batch should have only one example

        input_ = get_input_from_batch_wrapper(batch, use_cuda)

        enc_batch = input_['enc_batch']
        enc_lens = input_['enc_lens']

        result = defaultdict(list)
        for step, (token,
                   log_prob,
                   state,
                   context,
                   coverage) in enumerate(conditioning_summary.iterator()):

            for experiment in trange(num_experiments):
                encoder_result = self.model.encoder(enc_batch, enc_lens)
                encoder_outputs, encoder_feature, encoder_hidden = encoder_result

                # TODO Is this what it supposed to be?
                latest_tokens = [token for _ in range(config.beam_size)]

                y_t_1 = Variable(torch.LongTensor(latest_tokens))
                if use_cuda:
                    y_t_1 = y_t_1.cuda()

                state_h, state_c = state
                context = context
                all_state_h = [state_h for _ in range(config.beam_size)]
                all_state_c = [state_c for _ in range(config.beam_size)]
                all_context = [context for _ in range(config.beam_size)]

                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                        torch.stack(all_state_c, 0).unsqueeze(0))
                c_t_1 = torch.stack(all_context, 0)

                output = self.model.decoder(
                    y_t_1=y_t_1,
                    s_t_1=s_t_1,
                    encoder_outputs=encoder_outputs,
                    encoder_feature=encoder_feature,
                    enc_padding_mask=input_['enc_padding_mask'],
                    c_t_1=c_t_1,
                    extra_zeros=input_['extra_zeros'],
                    enc_batch_extend_vocab=input_['enc_batch_extend_vocab'],
                    coverage=None,
                    step=step)

                final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = output
                # log_probs = torch.log(final_dist)
                result[step].append(final_dist)

        stacked_result = defaultdict(int)
        for k, distributions in result:
            stacked_result[k] = torch.stack(distributions, 0)

        return stacked_result

    def batches(self):
        batch = self.batcher.next_batch()
        while batch is not None:
            yield batch
            batch = self.batcher.next_batch()

    @torch.no_grad()
    def run_experiments(self, num_experiments):
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        final_results = []
        for i, batch in enumerate(self.batches()):
            conditioning_summary = self.beam_search(batch)
            result = self.run_bayesian_dropout(batch, conditioning_summary,
                                               num_experiments)
            final_results.append(batch, result)

        return final_results


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, default=None)
    parser.add_argument('-n', '--num_experiments', required=False, type=str,
                        default=100)

    parser.add_argument('-d', '--dont_use_gpu', action='store_true')

    args = parser.parse_args()

    print('Used arguments: ', args)

    return args


def main():
    global use_cuda
    args = parse_arguments()

    model_file_path = args.model
    num_experiments = args.num_experiments

    bayesian_dropout = BayesianDropout(model_file_path)
    bayesian_dropout.run_experiments(num_experiments=num_experiments)


if __name__ == "__main__":
    main()

