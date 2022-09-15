import os

import numpy as np

from agents.mhb_baseline.nlp_model.agent import get_dialog, GridPredictor
from agents.mhb_baseline.nlp_model.utils import plot_voxel
from evaluator.iglu_evaluator import IGLUMetricsTracker


def compute_metric(grid, subtask):
    igm = IGLUMetricsTracker(None, subtask, {})
    return igm.get_metrics({'grid': grid})


def main():
    grid_predictor = None

    # os.environ['IGLU_DATA_PATH'] = 'iglu_data'

    from gridworld.tasks import Task

    from gridworld.data import IGLUDataset
    dataset = IGLUDataset(task_kwargs=None, force_download=False, )

    total_score = []

    for j, (task_id, session_id, subtask_id, subtask) in enumerate(dataset):
        str_id = str(task_id) + '-session-' + str(session_id).zfill(3) + '-subtask-' + str(subtask_id).zfill(3)
        print('Starting task:', str_id)
        subtask: Task = subtask

        if grid_predictor is None:
            grid_predictor = GridPredictor()

        dialog = get_dialog(subtask)
        predicted_grid = grid_predictor.predict_grid(dialog)

        if not os.path.exists('plots'):
            os.makedirs('plots')

        f1_score = round(compute_metric(predicted_grid, subtask)['completion_rate_f1'], 3)
        results = {'F1': f1_score}
        total_score.append(f1_score)
        results_str = " ".join([f"{metric}: {value}" for metric, value in results.items()])
        plot_voxel(predicted_grid, text=str_id + ' ' + f'({results_str})' + "\n" + dialog).savefig(
            f'./plots/{str_id}-predicted.png')
        plot_voxel(subtask.target_grid, text=str_id + " (Ground truth)\n" + dialog).savefig(
            f'./plots/{str_id}-gt.png')

    print('Total F1 score:', np.mean(total_score))


if __name__ == '__main__':
    main()
