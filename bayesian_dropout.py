# Except for the pytorch part content of this file
# is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import trange, tqdm
from itertools import islice
from pathlib import Path

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

    def iterator(self, stop=None, start=0):
        return islice(zip(self.tokens,
                          self.log_probs,
                          self.states,
                          self.contexts,
                          self.coverages), start, stop)

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
    def __init__(self, model_file_path, output_dir, use_elmo=False):

        self.output_dir = output_dir

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                          batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        # we use training mode, so dropout is activated
        self.model = Model(self.vocab,
                           model_file_path,
                           is_eval=False,
                           use_elmo=use_elmo,
                           use_cuda=use_cuda)

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
                    token_id = topk_ids[i, j].item()
                    token_id = 1 if token_id >= self.vocab.size() else token_id
                    new_beam = h.extend(token=token_id,
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
        self,
        batch,
        conditioning_summary,
        num_experiments,
        max_sentence_length=sys.maxsize,
        start=0,
        stop=100,
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
                   coverage) in enumerate(
                       tqdm(conditioning_summary.iterator(),
                            total=len(conditioning_summary.tokens))
                   ):

            for experiment in trange(num_experiments):
                print("Experiment:", experiment)
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
        for k, distributions in result.items():
            stacked_result[k] = torch.stack(distributions, 0)


        return stacked_result

    def batches(self):
        batch = self.batcher.next_batch()
        while batch is not None:
            yield batch
            batch = self.batcher.next_batch()

    def save_result(self, i, result):
        save_path = Path(self.output_dir) / (str(i) + '.pt')
        torch.save(result, save_path)

    @torch.no_grad()
    def run_experiments(self,
                        num_experiments,
                        max_num_summaries,
                        max_sentence_length=sys.maxsize,
                        start=0,
                        num_examples=20):

        for i, batch in enumerate(islice(self.batches(), start, start+num_examples), start):
            if i > max_num_summaries + start:
                break

            conditioning_summary = self.beam_search(batch)
            result = self.run_bayesian_dropout(batch, conditioning_summary,
                                            num_experiments,
                                            max_sentence_length=max_sentence_length,
                                            start=start)

            self.save_result(i, result)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, default=None,
                        help='Model file path')

    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None,
                        help='Output path for saved probabilities')

    parser.add_argument('-n', '--num_experiments', required=False, type=int,
                        default=100,
                        help="How many different outputs we would like to get for the same input")

    parser.add_argument('-s', '--max_num_summaries', required=False, type=int,
                        default=15000,
                        help="Run the bayesian dropout on this many examples only")

    parser.add_argument('-b', '--beginning', required=False, type=int,
                        default=0,
                        help="Begin with this summary, not the first one.")

    parser.add_argument('-l', '--max_sentence_length', required=False, type=int,
                        default=sys.maxsize,
                        help="Only for testing")

    parser.add_argument('-d', '--dont_use_gpu', action='store_true',
                        help="This flag will try to disable GPU usage")
    parser.add_argument("-e", "--use_elmo", required=False, action='store_true',
                        help="Use Glove+Elmo embeddings together, otherwise only Glove")

    args = parser.parse_args()

    print('Used arguments: ', args)

    return args


def main():
    global use_cuda
    args = parse_arguments()

    model_file_path = args.model
    output_dir = args.output_dir
    num_experiments = args.num_experiments
    dont_use_gpu = args.dont_use_gpu
    use_elmo = args.use_elmo
    max_num_summaries = args.max_num_summaries
    max_sentence_length = args.max_sentence_length
    beginning = args.beginning

    # experiment_name = '_'.join(os.basename(model_file_path).split('_')[1:])
    model_file_basepath = Path(model_file_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dont_use_gpu:
        use_cuda = False
        config.use_gpu = False

    bayesian_dropout = BayesianDropout(model_file_path, use_elmo=use_elmo, output_dir=output_dir)
    bayesian_dropout.run_experiments(num_experiments=num_experiments,
                                     max_num_summaries=max_num_summaries,
                                     max_sentence_length=max_sentence_length,
                                     start=beginning)


if __name__ == "__main__":
    main()

