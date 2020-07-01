import importlib
import os
import sys

import exputils
import imageio
import numpy as np
import torch

import autodisc as ad
from goalrepresent.datasets.image.imagedataset import LENIADataset


def collect_recon_loss_test_datasets(explorer):
    
    statistic = dict()
    
    test_dataset_idx = 0
    for test_dataset in test_datasets:
        statistic["data_{:03d}".format(test_dataset_idx)] = dict()

        reconstructions_root = np.ones(test_dataset.n_images) * -1
        reconstructions_leaf = np.ones(test_dataset.n_images) * -1
        
        model = explorer.visual_representation.model
        model.eval()
        with torch.no_grad():
            for test_data_idx in range(test_dataset.n_images):
                test_data = test_dataset.__getitem__(test_data_idx)["obs"].unsqueeze(0)
                all_nodes_outputs = model.network.depth_first_forward_whole_branch_preorder(test_data)
                for node_idx in range(len(all_nodes_outputs)):
                    cur_node_path = all_nodes_outputs[node_idx][0][0]
                    if cur_node_path == '0':
                        node_outputs = all_nodes_outputs[node_idx][2]
                        loss_inputs = {key: node_outputs[key] for key in model.loss_f.input_keys_list}
                        node_losses = model.loss_f(loss_inputs)
                        recon_loss = node_losses["recon"].item()
                        reconstructions_root[test_data_idx] = recon_loss
                    elif cur_node_path in model.network.get_leaf_pathes():
                        node_outputs = all_nodes_outputs[node_idx][2]
                        loss_inputs = {key: node_outputs[key] for key in model.loss_f.input_keys_list}
                        node_losses = model.loss_f(loss_inputs)
                        recon_loss = node_losses["recon"].item()
                        reconstructions_leaf[test_data_idx] = recon_loss
                    else:
                        pass


    
        statistic["data_{:03d}".format(test_dataset_idx)]['recon_loss_root'] = reconstructions_root
        
        statistic["data_{:03d}".format(test_dataset_idx)]['recon_loss_leaf'] = reconstructions_leaf

        test_dataset_idx += 1
        
    return statistic


# def collect_goal_space_representations_test_datasets(explorer):
#
#     statistic = dict()
#
#     test_dataset_idx = 0
#     for test_dataset in test_datasets:
#         statistic["data_{:03d}".format(test_dataset_idx)] = dict()
#
#         data = {}
#
#
#         model = explorer.visual_representation.model
#         for path in model.network.get_node_pathes():
#             if "gs_{}".format(path) not in statistic["data_{:03d}".format(test_dataset_idx)]:
#                 data["gs_{}".format(path)] = None
#
#         model.eval()
#         with torch.no_grad():
#             for path in model.network.get_node_pathes():
#                 cur_representation = []
#
#                 for test_data_idx in range(test_dataset.n_images):
#                     test_data = test_dataset.__getitem__(test_data_idx)["obs"].unsqueeze(0)
#                     representation = model.calc_embedding(test_data, path)
#                     cur_representation.append(representation.squeeze().cpu().detach().numpy())
#
#                 data["gs_{}".format(path)] = np.stack(cur_representation)
#
#         for k,v in data.items():
#
#             if len(np.shape(v)) == 1:
#                 v = np.array([v])
#             else:
#                 v = np.array(v)
#
#             statistic["data_{:03d}".format(test_dataset_idx)][k] = v
#
#
#         test_dataset_idx += 1
#
#     return statistic

def collect_final_observation(explorer):

    data = dict()

    for run_data in explorer.data:

        if run_data.observations is not None and len(run_data.observations.states) > 0:
            obs = run_data.observations
        else:
            [obs, statistics] = explorer.system.run(run_parameters=run_data.run_parameters,
                                                         stop_conditions=explorer.config.stop_conditions)

        # rescale values from [0 1] to [0 255] and convert to uint8 for saving as bw image
        img_data = obs.states[-1] * 255
        img_data = img_data.astype(np.uint8)

        png_image = imageio.imwrite(
                        imageio.RETURN_BYTES,
                        img_data,
                        format='PNG-PIL')

        data['{:06d}.png'.format(run_data.id)] = png_image

    return data


def collect_representation(explorer):
    data = dict()

    model = explorer.visual_representation.model
    if hasattr(model, "eval"):
        model.eval()

    if "ProgressiveTree" in model.__class__.__name__:

        all_nodes = model.network.get_node_pathes()
        n_latents = model.config.network.parameters.n_latents

        for node_path in all_nodes:
            data["gs_{}".format(node_path)] = []

        with torch.no_grad():

            for run_data in explorer.data:
                obs = run_data.observations
                if obs is None:
                    [obs, statistics] = explorer.system.run(run_parameters=run_data.run_parameters,
                                                             stop_conditions=explorer.config.stop_conditions)
                x = torch.from_numpy(obs["states"][-1]).float().unsqueeze(0).unsqueeze(0)
                x = model.push_variable_to_device(x)
                all_nodes_outputs = model.network.depth_first_forward_whole_branch_preorder(x)
                for node_idx in range(len(all_nodes_outputs)):
                    cur_node_path = all_nodes_outputs[node_idx][0][0]
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    cur_gs_representation = cur_node_outputs["z"].squeeze().detach().cpu().numpy()
                    data["gs_{}".format(cur_node_path)].append(cur_gs_representation)


        for node_path in all_nodes:
            data["gs_{}".format(node_path)] = np.stack(data["gs_{}".format(node_path)], axis=0)

    else:

        n_latents = model.config.network.parameters.n_latents
        data["gs_0"] = []

        with torch.no_grad():

            for run_data in explorer.data:
                obs = run_data.observations
                if obs is None:
                    [obs, statistics] = explorer.system.run(run_parameters=run_data.run_parameters,
                                                             stop_conditions=explorer.config.stop_conditions)
                x = torch.from_numpy(obs["states"][-1]).float().unsqueeze(0).unsqueeze(0)
                if hasattr(model, "push_variable_to_device"):
                    x = model.push_variable_to_device(x)
                z = model.calc_embedding(x).squeeze().detach().cpu().numpy()
                data["gs_0"].append(z)


        data["gs_0"] = np.stack(data["gs_0"], axis=0)



    return data


def collect_ids_per_node(explorer):
    data = dict()

    model = explorer.visual_representation.model
    model.eval()
    with torch.no_grad():
        for path in model.network.get_node_pathes():
            
            cur_gs_ids = []
            
            for run_data in explorer.data:
                obs = run_data.observations
                if obs is None:
                    [obs, statistics] = explorer.system.run(run_parameters=run_data.run_parameters,
                                                             stop_conditions=explorer.config.stop_conditions)
                x = torch.from_numpy(obs["states"][-1]).float().unsqueeze(0).unsqueeze(0)
                cur_representation = model.calc_embedding(x, path).squeeze()
                if not torch.isnan(cur_representation.sum()):
                    cur_gs_ids.append(run_data.id)
            
            data["gs_{}".format(path)] = np.stack(cur_gs_ids)

    return data


def load_explorer(experiment_directory):

    # load the full explorer without observations and add its config
    sys.path.append(experiment_directory)
    explorer = ad.explorers.GoalSpaceExplorer.load_explorer(os.path.join(experiment_directory, 'results'), run_ids=[], map_location='cpu', load_observations=False, verbose=False)
    explorer.data.config.load_observations = True
    explorer.data.config.memory_size_observations = 1

    spec = importlib.util.spec_from_file_location('experiment_config', os.path.join(experiment_directory, 'experiment_config.py'))
    experiment_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_config_module)
    explorer.config = experiment_config_module.get_explorer_config()
    
    # load test datasets
    global test_datasets
    test_datasets = []
    test_dataset_config = ad.Config()
    test_dataset_config.data_root = "/gpfswork/rech/zaj/ucf28eq/data/lenia_datasets/data_005/"
    test_dataset_config.split = "train"
    test_dataset_autodisc = LENIADataset(config = test_dataset_config)
    test_datasets.append(test_dataset_autodisc)
    return explorer


if __name__ == '__main__':

    experiments = '.'

    statistics = [#('final_observation', collect_final_observation, 'zip'),
                  #('representations', collect_representation),
                  #('ids_per_node', collect_ids_per_node),
                  ('recon_loss_test_datasets', collect_recon_loss_test_datasets),
                  #('goal_space_representations_test_datasets', collect_goal_space_representations_test_datasets),
                  ]

    exputils.calc_experiment_statistics(statistics, load_explorer, experiments, recalculate_statistics=False, verbose=True)
