import os

import exputils
import numpy as np
import torch

import autodisc as ad
from goalrepresent.models import BCBVAEModel, BCPatchBVAEModel, BCLeniaStatisticsModel, BCEllipticalFourierModel, \
    BCSpectrumFourierModel

BC_bvae_model = BCBVAEModel()
BC_patchbvae_model = BCPatchBVAEModel()
BC_leniastatistics_model = BCLeniaStatisticsModel()
BC_ellipticalfourier_model = BCEllipticalFourierModel()
BC_spectrumfourier_model = BCSpectrumFourierModel()

def calc_BC_bvae_analytic_space_representations(repetition_data):

    # load representation
    data = None

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            x = torch.from_numpy(run_data.observations["states"][-1]).unsqueeze(0).unsqueeze(0)
            cur_representation = BC_bvae_model.calc_embedding(x).squeeze().cpu().numpy()
            cur_rep_data.append(cur_representation)
            
        if data is None:
            data = np.expand_dims(np.stack(cur_rep_data), axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(np.stack(cur_rep_data), axis=0)], axis=0)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic["data"] = data

    return statistic


def calc_BC_patchbvae_analytic_space_representations(repetition_data):
    # load representation
    data = None

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            x = torch.from_numpy(run_data.observations["states"][-1]).unsqueeze(0).unsqueeze(0)
            cur_representation = BC_patchbvae_model.calc_embedding(x).squeeze().cpu().numpy()
            cur_rep_data.append(cur_representation)

        if data is None:
            data = np.expand_dims(np.stack(cur_rep_data), axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(np.stack(cur_rep_data), axis=0)], axis=0)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic["data"] = data

    return statistic


def calc_BC_leniastatistics_analytic_space_representations(repetition_data):
    # load representation
    data = None

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            x = torch.from_numpy(run_data.observations["states"][-1]).unsqueeze(0).unsqueeze(0)
            cur_representation = BC_leniastatistics_model.calc_embedding(x).squeeze().cpu().numpy()
            cur_rep_data.append(cur_representation)

        if data is None:
            data = np.expand_dims(np.stack(cur_rep_data), axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(np.stack(cur_rep_data), axis=0)], axis=0)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic["data"] = data

    return statistic


def calc_BC_ellipticalfourier_analytic_space_representations(repetition_data):
    # load representation
    data = None

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            x = torch.from_numpy(run_data.observations["states"][-1]).unsqueeze(0).unsqueeze(0)
            cur_representation = BC_ellipticalfourier_model.calc_embedding(x).squeeze().cpu().numpy()
            cur_rep_data.append(cur_representation)

        if data is None:
            data = np.expand_dims(np.stack(cur_rep_data), axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(np.stack(cur_rep_data), axis=0)], axis=0)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic["data"] = data

    return statistic


def calc_BC_spectrumfourier_analytic_space_representations(repetition_data):
    # load representation
    data = None

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            x = torch.from_numpy(run_data.observations["states"][-1]).unsqueeze(0).unsqueeze(0)
            cur_representation = BC_spectrumfourier_model.calc_embedding(x).squeeze().cpu().numpy()
            cur_rep_data.append(cur_representation)

        if data is None:
            data = np.expand_dims(np.stack(cur_rep_data), axis=0)
        else:
            data = np.concatenate([data, np.expand_dims(np.stack(cur_rep_data), axis=0)], axis=0)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic["data"] = data

    return statistic




def load_data(repetition_directories):

    data = dict()

    for repetition_directory in sorted(repetition_directories):

        # get id of the repetition from its foldername
        numbers_in_string = [int(s) for s in os.path.basename(repetition_directory).split('_') if s.isdigit()]
        repetition_id = numbers_in_string[0]

        # load the full explorer without observations and add its config
        datahandler_config = ad.ExplorationDataHandler.default_config()
        datahandler_config.memory_size_observations = 1

        rep_data = ad.ExplorationDataHandler.create(config=datahandler_config, directory=os.path.join(repetition_directory, 'results'))
        rep_data.load(load_observations=False, verbose=True)

        data[repetition_id] = rep_data

    return data


if __name__ == '__main__':

    experiments = '.'

    statistics = [ ('BC_bvae_analytic_space_representations', calc_BC_bvae_analytic_space_representations),
                   ('BC_patchbvae_analytic_space_representations', calc_BC_patchbvae_analytic_space_representations),
                   ('BC_leniastatistics_analytic_space_representations', calc_BC_leniastatistics_analytic_space_representations),
                   ('BC_ellipticalfourier_analytic_space_representations', calc_BC_ellipticalfourier_analytic_space_representations),
                   ('BC_spectrumfourier_analytic_space_representations', calc_BC_spectrumfourier_analytic_space_representations),
                   ]

    exputils.calc_statistics_over_repetitions(statistics, load_data, experiments, recalculate_statistics=False, verbose=True)
