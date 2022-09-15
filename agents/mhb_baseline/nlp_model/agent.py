import os

import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

from agents.mhb_baseline.nlp_model.utils import update_state_from_action, logging, parse_logs

script_dir = os.path.dirname(os.path.realpath(__file__))


def load_model(args):
    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(script_dir, "tokenizer/"),
                                            max_source_length=args.max_source_length,
                                            max_target_length=args.max_target_length,
                                            model_max_length=args.max_source_length)
    # tokenizer.save_vocabulary(".")

    special_tokens_dict = {'additional_special_tokens': ['<Architect>', '<Builder>', '<sep1>']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # model
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(script_dir, "model/"))
    # model.save_pretrained("model")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    return model, tokenizer


def generate_actions(args, model, tokenizer, command, history, log_file=None):
    context = parse_logs(history) + '<Architect> ' + command.strip()

    input_ids = tokenizer(f"{args.task_prefix} {context}", return_tensors="pt").input_ids

    outputs = model.generate(input_ids.cuda(), min_length=2, max_length=args.max_target_length, )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if args.verbose > 0:
        logging(log_file, f'model input: {context}')
        logging(log_file, f'model output: {prediction}')

    return prediction


def update_state(args, last_state, actions, log_file=None):
    if args.init_block:
        init_block = tuple(args.init_block)
    else:
        init_block = (5, 0, 5)

    n_actions, valid_actions = 0, 0

    actions = actions.split('.')
    for action in actions:
        if action:
            n_actions += 1
            last_state, ok = update_state_from_action(args, last_state, action, init_block, log_file)
            if ok:
                valid_actions += 1

    return last_state, {'n_actions': n_actions, 'n_valid': valid_actions}


def predict_voxel(command, model, tokenizer, history, voxel, args):
    actions = generate_actions(args, model, tokenizer, command, history, None)
    voxel, new_stats = update_state(args, voxel, actions, None)
    history.append((command, actions))
    right_voxel = np.transpose(voxel, (1, 0, 2))

    return history, right_voxel, voxel


def init_models(args):
    model, tokenizer = load_model(args)
    history = []
    voxel = np.zeros((11, 9, 11))
    stats = {'n_actions': 0, 'n_valid': 0}
    return model, tokenizer, history, stats, voxel


class DefArgs():
    def __init__(self):
        # self.model = "agents/mhb_baseline/nlp_model/t5-autoregressive-history-3-best.pt"
        self.model = os.path.join(script_dir, "t5-autoregressive-history-3-best.pt")
        self.out_dir = None
        self.verbose = 0
        self.logs_path = None
        self.histrory_len = None
        self.max_source_length = 512
        self.max_target_length = 128
        self.init_block = [5, 0, 5]
        self.task_prefix = 'implement given instructions: '


def coord_to_voxel(coordinates):
    voxel = np.zeros((11, 9, 11))
    for x, y, z, color in coordinates:
        voxel[x, y, z] = color
    return voxel


class FakeGridPredictor:
    def predict_grid(self, dialog, initial_grid=None):
        result = np.zeros((9, 11, 11))
        # result[0, 5, 5] = 2
        # result[1, 5, 5] = 2
        # result[2, 5, 5] = 2
        result[0, 5, 5] = 6
        result[0, 4, 5] = 6
        result[0, 3, 5] = 6
        result[0, 2, 5] = 6
        result[0, 1, 5] = 6
        return result


class GridPredictor:
    def __init__(self):
        self.args = DefArgs()
        self.model, self.tokenizer = load_model(self.args)

    def predict_grid(self, dialog, initial_grid=None):
        if dialog.count('Architect') >= 2:
            # if there is more than one command, we will use the last one
            dialog = dialog.split('Architect')[-1]

        if initial_grid is None:
            initial_grid = np.zeros((11, 9, 11))
        history = []
        _, grid, _ = predict_voxel(dialog, self.model, self.tokenizer, history, initial_grid, self.args)

        # fix colors
        self.remap_colors(grid)

        return grid

    @staticmethod
    def get_most_frequent_color(dialog):
        colors = {color: 0 for color in ['blue', 'green', 'red', 'orange', 'purple', 'yellow']}

        for color in colors.keys():
            colors[color] = dialog.count(color)
        return max(colors, key=colors.get)

    @staticmethod
    def remap_colors(v):
        remapping = {
            1: 3,
            2: 4,
            3: 6,
            4: 2,
            5: 1,
            6: 5,
        }

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in range(v.shape[2]):
                    if v[i, j, k] in remapping:
                        v[i, j, k] = remapping[v[i, j, k]]
