import os
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

import autodisc as ad
import goalrepresent as gr
from autodisc.explorers import helperprogressiveexplorer
from goalrepresent.datasets.image.imagedataset import LENIADataset


class ProgressiveExplorer(ad.core.Explorer):
    '''
    Goal space explorer that allows multiple goal spaces between which the process changes for its selection of goals.

    Source policies for a new exploration
    -------------------------------------

    config.source_policy_selection
        .type: 'optimal' or 'random'.
                Optimal selects a previous exploration which has the closest point in the goal space to the new goal.
                Random selects a random previous exploration as source.

        .constraints: Can be used to define a constraints on the source policies based on filters.
                      Is a list of filters or dictionaries with the following properties:
                            active: Defines if and when a constraint is active.
                                    Can be 'True', 'False', or a condition.
                            filter: Definition of the filter that defines which previous exploration runs are allowed as source.

    Examples of constraints:helperprogressiveexplorer.

        Only explorations for which statistics.classifier_animal is True:
            dict(active = True,
                 filter = ('statistics.classifier_animal', '==', True))

        Only explorations for which statistics.is_dead is False and statistics.classifier_animal is False:
            dict(active = True,
                 filter = (('statistics.is_dead', '==', False), 'and', ('statistics.classifier_animal', '==', False))

        Only active after 100 animals have been discovered:
            dict(active = (('sum', 'statistics.classifier_animal'), '>=', 100),
                 filter = (('statistics.is_dead', '==', False), 'and', ('statistics.classifier_animal', '==', False))
    '''


    @staticmethod
    def default_config():
        default_config = ad.core.Explorer.default_config()

        default_config.stop_conditions = 200
        default_config.num_of_random_initialization = 10

        default_config.run_parameters = []

        # representation 
        default_config.visual_representation = gr.representations.SingleModelRepresentation.default_config()
        
        # progressive growing
        default_config.progressive_growing = ad.Config()
        default_config.progressive_growing.split_trigger = {"active": False}
        
        # progressive_training
        default_config.progressive_training = ad.Config()
        default_config.progressive_training.n_runs_between_stages = 100
        default_config.progressive_training.n_epochs_per_stage = 40
        default_config.progressive_training.train_batch_size = 64
        default_config.progressive_training.importance_sampling_new_vs_old = 0.5
        default_config.progressive_training.dataset_constraints = []
        default_config.progressive_training.save_dataset = False
        
        # goal space selection
        default_config.goal_space_selection = ad.Config()
        default_config.goal_space_selection.type = 'random'  # either: 'random', 'probability_distribution', 'adaptive'
        default_config.goal_space_selection.measure = ad.Config()
        default_config.goal_space_selection.measure.update_constraints = []
        
        # goal selection 
        default_config.goal_selection = ad.Config()
        default_config.goal_selection.type = None # either: 'random', 'specific', 'function'
        
        # policy selection
        default_config.source_policy_selection = ad.Config()
        default_config.source_policy_selection.type = 'optimal' # either: 'optimal', 'random'
        default_config.source_policy_selection.constraints = []

        return default_config


    def __init__(self, system, datahandler=None, config=None, **kwargs):
        super().__init__(system=system, datahandler=datahandler, config=config, **kwargs)

        # check config
        ## check config goal_selection
        if self.config.goal_selection.type not in ['random', 'specific', 'function']:
            raise ValueError('Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection.type))

        # initialize goal_space_representation
        self.load_goal_space_representations()
        
        #if self.config.goal_space_selection.type not in ['random', 'probability_distribution', 'combination']
        self._init_goalspace_selection()

        self.policy_library = []
        self.goal_library = []

        self.statistics = ad.helper.data.AttrDict()
        self.statistics.target_goal_spaces = []
        self.statistics.target_goals = []
        self.statistics.reached_goals = []
        self.statistics.reached_initial_goals = []
        self.statistics.target_goal_space_measures = []
        self.statistics.target_goal_space_probabilities = []


    def load_goal_space_representations(self):
        
        self.visual_representation = gr.representations.SingleModelRepresentation(config=self.config.visual_representation)
        self.goal_space_node_pathes=["0"] # give root representation as initial goal space
            
    
    def _init_goalspace_selection(self):

        n_goal_spaces = len(self.goal_space_node_pathes)

        if self.config.goal_space_selection.type == 'random':
            self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureFixedProbabilities(probabilities=[1/n_goal_spaces]*n_goal_spaces)
            self.goalspace_selection_algo = helperprogressiveexplorer.GSSelectionAlgoFixedProbabilities()

        elif self.config.goal_space_selection.type == 'probability_distribution':
            self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureFixedProbabilities(probabilities=self.config.goal_space_selection.distribution)
            self.goalspace_selection_algo = helperprogressiveexplorer.GSSelectionAlgoFixedProbabilities()

        elif self.config.goal_space_selection.type == 'adaptive':

            n_goal_spaces = len(self.goal_space_node_pathes)

            if self.config.goal_space_selection.measure.type == 'score_per_goalspace':
                self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureScorePerGoalspace(
                    scores=[0] * n_goal_spaces)
            elif self.config.goal_space_selection.measure.type == 'diversity_per_goalspace':
                self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureDiversityPerGoalspace(
                    n_goal_spaces, self.config.goal_space_selection.measure)
            elif self.config.goal_space_selection.measure.type == 'mean_diversity_over_goalspaces':
                self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureMeanDiversityOverGoalspaces(
                    n_goal_spaces, self.config.goal_space_selection.measure)
            elif self.config.goal_space_selection.measure.type == 'diversity_over_combined_goalspaces':
                self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureDiversityOverCombinedGoalspaces(
                    n_goal_spaces, self.config.goal_space_selection.measure)
            elif self.config.goal_space_selection.measure.type == 'competence_per_goalspace':
                self.goalspace_selection_measure = helperprogressiveexplorer.GSSelectionMeasureCompetencePerGoalSpace(
                    self, n_goal_spaces, self.config.goal_space_selection.measure)
            else:
                raise ValueError('Unknown goal space selection measure type {!r} in the configuration!'.format(
                    self.config.goal_space_selection.measure.type))

            if self.config.goal_space_selection.selection_algo.type == 'epsilon_greedy':
                self.goalspace_selection_algo = ad.helper.sampling.EpsilonGreedySelection(self.config.goal_space_selection.selection_algo.epsilon)
            elif self.config.goal_space_selection.selection_algo.type == 'softmax':
                self.goalspace_selection_algo = ad.helper.sampling.SoftmaxSelection(
                    self.config.goal_space_selection.selection_algo.beta)
            elif self.config.goal_space_selection.selection_algo.type == 'epsilon_softmax':
                self.goalspace_selection_algo = ad.helper.sampling.EpsilonSoftmaxSelection(
                    self.config.goal_space_selection.selection_algo.epsilon,
                    self.config.goal_space_selection.selection_algo.beta)
            else:
                raise ValueError('Unknown goal space selection algo type {!r} in the configuration!'.format(
                    self.config.goal_space_selection.selection_algo.type))

        else:
            raise ValueError('Unknown goal space selection type {!r} in the configuration!'.format(
                self.config.goal_space_selection.type))

        self.goalspace_selection_measure_update_constraints = dict(active=False, filter=None)
        # active can be: True (ie updated with every data run) / "after_split" (ie updated after each split) / some condition on all the data (eg: after 1000 patterns have been discovered)
        if self.config.goal_space_selection.measure.update_constraints:
            self.goalspace_selection_measure_update_constraints = ad.config.set_default_config(
                self.config.goal_space_selection.measure.update_constraints,
                self.goalspace_selection_measure_update_constraints)
            
    def get_next_goal_space_idx(self):
        '''Defines the next goal space that should be used to select a source policy.'''
        measure_values = self.goalspace_selection_measure.get_measure()
        goal_space_idx, probabilities = self.goalspace_selection_algo.do_selection(measure_values)

        return goal_space_idx, measure_values, probabilities



    def get_next_goal(self, goal_space_idx):
        '''Defines the next goal of the exploration.'''

        goal_selection_config = self.config.goal_selection
                
        if goal_selection_config.type == 'random':

            if hasattr(goal_selection_config, 'sampling'):
                target_goal = np.zeros(len(goal_selection_config.sampling))
                for idx, sampling_config in enumerate(goal_selection_config.sampling):
                    target_goal[idx] = ad.helper.sampling.sample_value(self.random, sampling_config)

            elif hasattr(goal_selection_config, 'sampling_from_reached_boundaries'):
                target_goal = np.zeros(self.goal_library_extent[goal_space_idx].shape[0])
                margin_min = 0
                margin_max = 0
                if 'margin_min' in goal_selection_config.sampling_from_reached_boundaries:
                    margin_min = float(goal_selection_config.sampling_from_reached_boundaries['margin_min'])
                if 'margin_max' in goal_selection_config.sampling_from_reached_boundaries:
                    margin_max = float(goal_selection_config.sampling_from_reached_boundaries['margin_max'])
                for latent_idx in range(self.goal_library_extent[goal_space_idx].shape[0]):
                    curr_interval_width = self.goal_library_extent[goal_space_idx][latent_idx,1] - self.goal_library_extent[goal_space_idx][latent_idx,0]
                    curr_min_val = self.goal_library_extent[goal_space_idx][latent_idx,0] - margin_min * curr_interval_width
                    curr_max_val = self.goal_library_extent[goal_space_idx][latent_idx,1] + margin_max * curr_interval_width
                    target_goal[latent_idx] = ad.helper.sampling.sample_value(self.random, (curr_min_val, curr_max_val))
            else:
                raise ValueError('random interval must be either ["sampling","sampling_from_reached_boundaries"] in {!r}'.format(self.config.goal_selection))

        elif goal_selection_config.type == 'specific':

            if np.ndim(goal_selection_config.goals) == 1:
                target_goal = np.array(goal_selection_config.goals)
            else:
                rand_idx = self.random.randint(np.shape(goal_selection_config.goals)[0])
                target_goal = np.array(goal_selection_config.goals[rand_idx])

        elif goal_selection_config.type == 'function':
            pass

        else:
            raise ValueError('Unknown goal generation type {!r} in the configuration!'.format(goal_selection_config.type))

        return target_goal


    def get_possible_source_policies_inds(self, goal_space_idx):

        possible_run_inds = np.full(np.shape(self.goal_library[goal_space_idx])[0], True)

        # apply constraints on the possible source policies if defined under config.source_policy_selection.constraints
        if self.config.source_policy_selection.constraints:

            for constraint in self.config.source_policy_selection.constraints:

                if isinstance(constraint, tuple):
                    # if tuple, then this is the contraint and it is active
                    cur_is_active = True
                    cur_filter = constraint
                else:
                    # otherwise assume it is a dict/config with the fields: active, filter
                    if 'active' not in constraint:
                        cur_is_active = True
                    else:
                        if isinstance(constraint['active'], tuple):
                            cur_is_active = self.data.filter(constraint['active'])
                        else:
                            cur_is_active = constraint['active']

                    cur_filter = constraint['filter']

                if cur_is_active:
                    possible_run_inds = possible_run_inds & self.data.filter(cur_filter)

        if np.all(possible_run_inds == False):
            warnings.warn('No policy fullfilled the constraint. Allow all policies.')
            possible_run_inds = np.full(np.shape(self.goal_library[goal_space_idx])[0], True)

        return possible_run_inds


    def get_source_policy_idx(self, target_goal, goal_space_idx, goal_space_extent=None):

        possible_run_inds = self.get_possible_source_policies_inds(goal_space_idx)

        possible_run_idxs = np.array(list(range(np.shape(self.goal_library[goal_space_idx])[0])))
        possible_run_idxs = possible_run_idxs[possible_run_inds]

        if self.config.source_policy_selection.type == 'optimal':

            # get distance to other goals
            goal_distances = self.visual_representation.calc_distance(target_goal, self.goal_library[goal_space_idx][
                                                                                   possible_run_inds, :],
                                                                      goal_space_extent)

            # select goal with minimal distance
            try:
                min_values = np.where(goal_distances == np.nanmin(goal_distances))[0]
                min_val = np.random.choice(min_values)
                source_policy_idx = possible_run_idxs[min_val]
            except:
                source_policy_idx = np.random.choice(possible_run_idxs)
            # try:
            #     source_policy_idx = possible_run_idxs[np.nanargmin(goal_distances)]
            # except:
            #     source_policy_idx = np.random.choice(possible_run_idxs)

        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = possible_run_idxs[np.random.randint(len(possible_run_idxs))]
        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(self.config.source_policy_selection.type))

        return source_policy_idx


    def run(self, runs, verbose=True, continue_existing_experiment=False):

        if isinstance(runs, int):
            runs = list(range(runs))

        if not continue_existing_experiment:
            self.policy_library = []
            self.goal_library = [[] for _ in range(len(self.goal_space_node_pathes))]
            self.goal_library_extent = [[] for _ in range(len(self.goal_space_node_pathes))] # store min an max for sampling goals

            self.statistics.target_goal_spaces = []
            self.statistics.target_goals = []
            self.statistics.reached_goals = []
            self.statistics.reached_initial_goals = []
            self.statistics.target_goal_space_measures = []
            self.statistics.target_goal_space_probabilities = []

            system_state_size = (self.system.system_parameters.size_y, self.system.system_parameters.size_x)

            # save a stack of last observations in memory to recompute new goals faster
            final_observations = []
            # save numerical labels for each run describing if they are {0: 'animal', 1: 'non_animal', -1: 'dead'}
            labels = []

            # prepare datasets
            dataset_config = gr.Config()
            dataset_config.img_size = system_state_size
            dataset_config.data_augmentation = self.config.progressive_training.dataset_augment
            train_dataset = LENIADataset(config=dataset_config)
            weights_train_dataset = [1.]
            weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train_dataset, 1)
            train_loader = DataLoader(train_dataset, batch_size=self.config.progressive_training.train_batch_size,
                                      sampler=weighted_sampler, num_workers=0)

            dataset_config.data_augmentation = False
            valid_dataset = LENIADataset(config=dataset_config)
            valid_loader = DataLoader(valid_dataset, self.config.progressive_training.train_batch_size, num_workers=0)

            # counters
            self.counters = ad.Config()
            self.counters.stage_idx = 0
            self.counters.last_run_idx_seen_by_nn = 0
            self.counters.n_train_dataset_curr_stage = 0
            self.counters.n_valid_dataset_curr_stage = 0

        else:

            assert len(self.data)>0, print("Please set continue_existing_experiment=False is explorer.data is empty")

            system_state_size = (self.system.system_parameters.size_y, self.system.system_parameters.size_x)
            # recreate final observations/labels from saved data
            final_observations = []
            labels = []
            for run_data in self.data:
                final_observation = torch.from_numpy(run_data.observations.states[-1]).float().unsqueeze(0)  # C*H*W FloatTensor
                final_observations.append(final_observation)
                # add label describing final observation
                if not run_data.statistics.is_dead:
                    if run_data.statistics.classifier_animal:
                        label = 0
                    else:
                        label = 1
                else:
                    label = -1
                labels.append(label)

            ''' initialize empty train/valid loader '''
            dataset_config = gr.Config()
            dataset_config.img_size = system_state_size
            dataset_config.data_augmentation = self.config.progressive_training.dataset_augment
            train_dataset = LENIADataset(config=dataset_config)
            weights_train_dataset = [1.]
            weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train_dataset, 1)
            train_loader = DataLoader(train_dataset, batch_size=self.config.progressive_training.train_batch_size,
                                      sampler=weighted_sampler, num_workers=0)

            dataset_config.data_augmentation = False
            valid_dataset = LENIADataset(config=dataset_config)
            valid_loader = DataLoader(valid_dataset, self.config.progressive_training.train_batch_size, num_workers=0)

            ''' Filter samples '''
            constrained_run_inds = np.full(np.shape(self.goal_library[0])[0], True)
            # apply constraints on the last runs to train network on a specific type of image
            if self.config.progressive_training.dataset_constraints:
                for constraint in self.config.progressive_training.dataset_constraints:
                    if isinstance(constraint, tuple):
                        # if tuple, then this is the contraint and it is active
                        cur_is_active = True
                        cur_filter = constraint
                    else:
                        # otherwise assume it is a dict/config with the fields: active, filter
                        if 'active' not in constraint:
                            cur_is_active = True
                        else:
                            if isinstance(constraint['active'], tuple):
                                cur_is_active = self.data.filter(constraint['active'])
                            else:
                                cur_is_active = constraint['active']
                        cur_filter = constraint['filter']
                    if cur_is_active:
                        constrained_run_inds = constrained_run_inds & self.data.filter(cur_filter)

            ''' recreate train/valid loader from saved data '''
            for run_id in constrained_run_inds:
                if run_id <= self.counters.last_run_idx_seen_by_nn:
                    if (train_loader.dataset.n_images + valid_loader.dataset.n_images + 1) % 10 == 0:
                        valid_loader.dataset.images = torch.cat(
                            [valid_loader.dataset.images, final_observations[run_id].unsqueeze(0)])
                        valid_loader.dataset.labels = torch.cat(
                            [valid_loader.dataset.labels, torch.LongTensor([labels[run_id]]).unsqueeze(0)])
                        valid_loader.dataset.n_images += 1
                    else:
                        train_loader.dataset.images = torch.cat(
                            [train_loader.dataset.images, final_observations[run_id].unsqueeze(0)])
                        train_loader.dataset.labels = torch.cat(
                            [train_loader.dataset.labels, torch.LongTensor([labels[run_id]]).unsqueeze(0)])
                        train_loader.dataset.n_images += 1


        # as we train more after the split we remove some training stages at the end for fairness
        n_max_stages = (len(runs) - 1) // self.config.progressive_training.n_runs_between_stages


        if verbose:
            counter = 0
            ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')

        for run_idx in runs:

            if run_idx not in self.data:

                # set the seed if the user defined one
                if self.config.seed is not None:
                    seed = 100000 * self.config.seed + run_idx
                    self.random.seed(seed)
                    random.seed(seed) # standard random is needed for the neat sampling process
                    np.random.seed(seed)  # Numpy module
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed) # if using gpu
                    torch.cuda.manual_seed_all(seed)  # if  using multi-GPU
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                else:
                    seed = None

                target_goal_space_idx = np.nan
                target_goal = []
                
                target_goal_space_measures = [np.nan] * len(self.goal_space_node_pathes)
                target_goal_space_probabilities = [np.nan] * len(self.goal_space_node_pathes)


                # get a policy - run_parameters
                policy_parameters = ad.helper.data.AttrDict()
                run_parameters = ad.helper.data.AttrDict()

                source_policy_idx = None

                # random sampling if not enough in library
                if len(self.policy_library) < self.config.num_of_random_initialization:
                    # initialize the parameters

                    for parameter_config in self.config.run_parameters:

                        if parameter_config.type == 'cppn_evolution':

                            cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(config=parameter_config['init'], matrix_size=system_state_size)
                            cppn_evo.do_evolution()

                            if parameter_config.init.best_genome_of_last_generation:
                                policy_parameter = cppn_evo.get_best_genome_last_generation()
                                run_parameter = cppn_evo.get_best_matrix_last_generation()

                            else:
                                policy_parameter = cppn_evo.get_best_genome()
                                run_parameter = cppn_evo.get_best_matrix()

                        elif parameter_config.type == 'sampling':

                            policy_parameter = ad.helper.sampling.sample_value(self.random, parameter_config['init'])
                            run_parameter = policy_parameter

                        else:
                            raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))

                        policy_parameters[parameter_config['name']] = policy_parameter
                        run_parameters[parameter_config['name']] = run_parameter

                else:

                    # get the goal space which should be responsible for the sampling of a goal
                    target_goal_space_idx, target_goal_space_measures, target_goal_space_probabilities = self.get_next_goal_space_idx()
                    # sample a goal space from the goal space
                    target_goal = self.get_next_goal(target_goal_space_idx)

                    # get source policy which should be mutated
                    source_policy_idx = self.get_source_policy_idx(target_goal, target_goal_space_idx, self.goal_library_extent[target_goal_space_idx])
                        
                    source_policy = self.policy_library[source_policy_idx]

                    for parameter_config in self.config.run_parameters:

                        if parameter_config.type == 'cppn_evolution':

                            cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(init_population=source_policy[parameter_config['name']],
                                                                           config=parameter_config['mutate'],
                                                                           matrix_size=system_state_size)
                            cppn_evo.do_evolution()

                            if parameter_config.init.best_genome_of_last_generation:
                                policy_parameter = cppn_evo.get_best_genome_last_generation()
                                run_parameter = cppn_evo.get_best_matrix_last_generation()
                            else:
                                policy_parameter = cppn_evo.get_best_genome()
                                run_parameter = cppn_evo.get_best_matrix()

                        elif parameter_config.type == 'sampling':

                            policy_parameter = ad.helper.sampling.mutate_value(val=source_policy[parameter_config['name']],
                                                                               rnd=self.random,
                                                                               config=parameter_config['mutate'])
                            run_parameter = policy_parameter

                        else:
                            raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))

                        policy_parameters[parameter_config['name']] = policy_parameter
                        run_parameters[parameter_config['name']] = run_parameter

                # run with parameters
                [observations, statistics] = self.system.run(run_parameters=run_parameters, stop_conditions=self.config.stop_conditions)

                # get goal-space of results
                reached_goal_per_space = []

                for goal_space_node_path in self.goal_space_node_pathes:
                    reached_goal = self.visual_representation.calc(observations, node_path=goal_space_node_path)
                    reached_goal_per_space.append(reached_goal)

                # save results
                self.data.add_run_data(id=run_idx,
                                       seed=seed,
                                       run_parameters=run_parameters,
                                       observations=observations,
                                       statistics=statistics,
                                       source_policy_idx=source_policy_idx,   # idx of the exploration that was used as source to generate the parameters for the current exploration
                                       target_goal_space_idx=target_goal_space_idx,
                                       target_goal=target_goal,
                                       reached_goal_per_space=reached_goal_per_space,)

                # add policy and reached goal into the libraries
                # do it after the run data is saved to not save them if there is an error during the saving
                self.policy_library.append(policy_parameters)

                for goal_space_idx, reached_goal in enumerate(reached_goal_per_space):
                    if len(self.goal_library[goal_space_idx]) <= 0:
                        self.goal_library[goal_space_idx] = np.array([reached_goal])
                        self.goal_library_extent[goal_space_idx] = np.stack([reached_goal, reached_goal], axis=-1)
                    else:
                        self.goal_library[goal_space_idx] = np.vstack([self.goal_library[goal_space_idx], reached_goal])
                        self.goal_library_extent[goal_space_idx][:, 0] = np.fmin(
                            self.goal_library_extent[goal_space_idx][:, 0], reached_goal)
                        self.goal_library_extent[goal_space_idx][:, 1] = np.fmax(
                            self.goal_library_extent[goal_space_idx][:, 1], reached_goal)

                # update the measure for the selection of the goal space
                ## if tuple: check if constraint is fullfilled
                ## if bool: True of False
                ## if string: only string possible is "after_split" so not considered here
                cur_goalspace_selection_measure_update = False
                if isinstance(self.goalspace_selection_measure_update_constraints["active"], tuple):
                    cur_goalspace_selection_measure_update = self.data.filter(
                        self.goalspace_selection_measure_update_constraints["active"])
                elif isinstance(self.goalspace_selection_measure_update_constraints["active"], bool):
                    cur_goalspace_selection_measure_update = self.goalspace_selection_measure_update_constraints[
                        "active"]
                if cur_goalspace_selection_measure_update:
                    add_data = True
                    cur_data = [self.data[run_idx]]
                    if not ad.helper.misc.do_filter_boolean(cur_data,
                                                            self.goalspace_selection_measure_update_constraints[
                                                                "filter"]):
                        add_data = False
                    if add_data:
                        reached_goal_per_space = [self.goal_library[gs_idx][run_idx] for gs_idx in
                                                  range(len(self.goal_space_node_pathes))]
                        self.goalspace_selection_measure.update_measure(self.data[run_idx].target_goal_space_idx,
                                                                        self.data[run_idx].target_goal,
                                                                        reached_goal_per_space,
                                                                        self.goal_library_extent)
                    

                # save statistics
                if len(target_goal) <= 0:
                    self.statistics.reached_initial_goals.append(reached_goal_per_space)
                else:
                    self.statistics.target_goal_spaces.append(target_goal_space_idx)
                    self.statistics.target_goals.append(target_goal)
                    self.statistics.reached_goals.append(reached_goal_per_space)
                
                self.statistics.target_goal_space_measures.append(target_goal_space_measures)
                self.statistics.target_goal_space_probabilities.append(target_goal_space_probabilities)
                    
                # add data to final_observations stack
                final_observation = torch.from_numpy(observations.states[-1]).float().unsqueeze(0) # C*H*W FloatTensor
                final_observations.append(final_observation)
                # add label describing final observation
                if not statistics.is_dead:
                    if statistics.classifier_animal:
                        label = 0
                    else:
                        label = 1
                else:
                    label = -1
                labels.append(label)

                if (run_idx + 1) == len(runs):
                    self.save()
                    break;

                ''' Train stage '''
                if ((run_idx + 1) % self.config.progressive_training.n_runs_between_stages == 0):
                    if self.counters.stage_idx < n_max_stages:
                        self.do_training_stage(train_loader, valid_loader,
                                               self.config.progressive_training.n_epochs_per_stage, final_observations,
                                               labels, run_idx)

                        ''' if split, update goal library'''
                        has_splitted = ('Progressive' in self.visual_representation.model.__class__.__name__) and \
                                       len(self.visual_representation.model.network.get_leaf_pathes()) != len(self.goal_space_node_pathes)
                        if has_splitted:
                            self.goal_space_node_pathes = self.visual_representation.model.network.get_leaf_pathes()
                            ''' Update goal library after split'''
                            goal_library_init = np.empty_like(self.goal_library[0])
                            goal_library_extent_init = np.empty_like(self.goal_library_extent[0])
                            goal_library_extent_init[:, 0] = + np.inf
                            goal_library_extent_init[:, 1] = - np.inf
                            self.goal_library = [deepcopy(goal_library_init) for _ in
                                                 range(len(self.goal_space_node_pathes))]
                            self.goal_library_extent = [deepcopy(goal_library_extent_init) for _ in
                                                        range(len(self.goal_space_node_pathes))]

                            n_runs = len(self.goal_library[0])
                            n_full_batches = n_runs // self.config.progressive_training.train_batch_size
                            n_remaining_idx = n_runs - n_full_batches * self.config.progressive_training.train_batch_size

                            self.visual_representation.model.eval()
                            with torch.no_grad():
                                for batch_idx in range(n_full_batches):
                                    x = torch.stack(final_observations[
                                                    (batch_idx * self.config.progressive_training.train_batch_size): ((
                                                                                                                              batch_idx + 1) * self.config.progressive_training.train_batch_size)],
                                                    dim=0)
                                    for goal_space_idx in range(len(self.goal_space_node_pathes)):
                                        output_goals = self.visual_representation.model.calc_embedding(x, node_path=
                                        self.goal_space_node_pathes[goal_space_idx])
                                        for idx in range(self.config.progressive_training.train_batch_size):
                                            reach_goal_updated = output_goals[idx].squeeze(0).cpu().numpy()
                                            self.goal_library[goal_space_idx][(
                                                                                      batch_idx * self.config.progressive_training.train_batch_size) + idx] = reach_goal_updated
                                            self.goal_library_extent[goal_space_idx][:, 0] = np.fmin(
                                                self.goal_library_extent[goal_space_idx][:, 0], reach_goal_updated)
                                            self.goal_library_extent[goal_space_idx][:, 1] = np.fmax(
                                                self.goal_library_extent[goal_space_idx][:, 1], reach_goal_updated)

                                if n_remaining_idx > 0:
                                    x = torch.stack(final_observations[(
                                                                               n_full_batches * self.config.progressive_training.train_batch_size):],
                                                    dim=0)
                                    for goal_space_idx in range(len(self.goal_space_node_pathes)):
                                        output_goals = self.visual_representation.model.calc_embedding(x, node_path=
                                        self.goal_space_node_pathes[goal_space_idx])
                                        for idx in range(n_remaining_idx):
                                            reach_goal_updated = output_goals[idx].squeeze(0).cpu().numpy()
                                            self.goal_library[goal_space_idx][(
                                                                                      n_full_batches * self.config.progressive_training.train_batch_size) + idx] = reach_goal_updated
                                            self.goal_library_extent[goal_space_idx][:, 0] = np.fmin(
                                                self.goal_library_extent[goal_space_idx][:, 0], reach_goal_updated)
                                            self.goal_library_extent[goal_space_idx][:, 1] = np.fmax(
                                                self.goal_library_extent[goal_space_idx][:, 1], reach_goal_updated)

                            ''' Update goal space selection after split'''
                            self._init_goalspace_selection()
                            if self.goalspace_selection_measure_update_constraints["active"] == "after_split":
                                for old_run in range(n_runs):
                                    add_data = True
                                    cur_data = [self.data[old_run]]
                                    if not ad.helper.misc.do_filter_boolean(cur_data,
                                                                            self.goalspace_selection_measure_update_constraints[
                                                                                "filter"]):
                                        add_data = False
                                    if add_data:
                                        updated_reached_goal_per_space = [self.goal_library[gs_idx][old_run] for
                                                                          gs_idx in
                                                                          range(len(self.goal_space_node_pathes))]
                                        self.goalspace_selection_measure.update_measure(
                                            self.data[old_run].target_goal_space_idx,
                                            self.data[old_run].target_goal,
                                            updated_reached_goal_per_space,
                                            self.goal_library_extent)

                            ''' Save explorer '''
                            self.save()

                            


                if verbose:
                    counter += 1
                    ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')
                    if counter == len(runs):
                        print('')

        # save statistics
        self.data.add_exploration_data(statistics=self.statistics)
        
        

    def do_training_stage(self, train_loader, valid_loader, n_epochs, all_observations, all_labels, run_idx):
        ''' Filter samples '''
        constrained_run_inds = np.full(np.shape(self.goal_library[0])[0], True)
        # apply constraints on the last runs to train network on a specific type of image
        if self.config.progressive_training.dataset_constraints:
            for constraint in self.config.progressive_training.dataset_constraints:
                if isinstance(constraint, tuple):
                    # if tuple, then this is the contraint and it is active
                    cur_is_active = True
                    cur_filter = constraint
                else:
                    # otherwise assume it is a dict/config with the fields: active, filter
                    if 'active' not in constraint:
                        cur_is_active = True
                    else:
                        if isinstance(constraint['active'], tuple):
                            cur_is_active = self.data.filter(constraint['active'])
                        else:
                            cur_is_active = constraint['active']
                    cur_filter = constraint['filter']
                if cur_is_active:
                    constrained_run_inds = constrained_run_inds & self.data.filter(cur_filter)
    
        if np.sum(constrained_run_inds) < 10:
            warnings.warn('Not enough runs fullfilled the constraint to start training, skipping the init stage')
            
        else:          
            ''' If enough samples start stage of training '''
            
            ''' Update training and validation datasets and weights '''
            last_constrained_run_idxs = np.array(list(range(self.counters.last_run_idx_seen_by_nn + 1, np.shape(self.goal_library[0])[0])))
            last_constrained_run_idxs = last_constrained_run_idxs[constrained_run_inds[self.counters.last_run_idx_seen_by_nn + 1:]]
            
            # iterate to new discoverues and add it to train/valid dataset
            run_ids_added_to_train_dataset = []
            run_ids_added_to_valid_dataset = []
            for run_id in last_constrained_run_idxs:
                if (train_loader.dataset.n_images + valid_loader.dataset.n_images + 1) % 10 == 0:
                    valid_loader.dataset.images = torch.cat(
                        [valid_loader.dataset.images, all_observations[run_id].unsqueeze(0)])
                    valid_loader.dataset.labels = torch.cat(
                        [valid_loader.dataset.labels, torch.LongTensor([all_labels[run_id]]).unsqueeze(0)])
                    valid_loader.dataset.n_images += 1
                    self.counters.n_valid_dataset_curr_stage += 1
                    run_ids_added_to_valid_dataset.append(run_id)
                else:
                    train_loader.dataset.images = torch.cat(
                        [train_loader.dataset.images, all_observations[run_id].unsqueeze(0)])
                    train_loader.dataset.labels = torch.cat(
                        [train_loader.dataset.labels, torch.LongTensor([all_labels[run_id]]).unsqueeze(0)])
                    train_loader.dataset.n_images += 1
                    self.counters.n_train_dataset_curr_stage += 1
                    run_ids_added_to_train_dataset.append(run_id)
            
            # update weight
            if self.counters.stage_idx == 0:
                weights = [1.0 / train_loader.dataset.n_images] * (train_loader.dataset.n_images)
                train_loader.sampler.num_samples = len(weights)
                train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)
            else:
                weights = [(1.0 - self.config.progressive_training.importance_sampling_new_vs_old) / (
                            train_loader.dataset.n_images - self.counters.n_train_dataset_curr_stage)] * (
                                  train_loader.dataset.n_images - self.counters.n_train_dataset_curr_stage)
                if self.counters.n_train_dataset_curr_stage > 0:
                    weights += ([self.config.progressive_training.importance_sampling_new_vs_old / (
                        self.counters.n_train_dataset_curr_stage)] * self.counters.n_train_dataset_curr_stage)
                train_loader.sampler.num_samples = len(weights)
                train_loader.sampler.weights = torch.tensor(weights, dtype=torch.double)

            ''' training stage ... '''
            training_config = self.visual_representation.config.training
            training_config.n_epochs = n_epochs
            training_config.split_trigger = self.config.progressive_growing.split_trigger
            training_config.alternated_backward = self.config.progressive_training.alternated_backward
            self.visual_representation.run_training(train_loader, valid_loader, training_config, keep_best_model=False,
                                                    logging=True)

            ''' save representation model after stage'''
            model_filepath = os.path.join(self.visual_representation.model.config.checkpoint.folder,
                                          'stage_{:06d}_weight_model.pth'.format(self.counters.stage_idx))
            self.visual_representation.model.save_checkpoint(model_filepath)

            ''' Update goal libary after the training stage'''
            goal_space_idx = 0
            for goal_space_node_path in self.goal_space_node_pathes:
                # in goal_library: goals are recomputed after each stage
                # in reached_goal (that is in run_data["reached_goal"]) we save the goals that are set in the current goal space at time t
                # we compute new goals by batches as for training to go faster
                n_old_runs = len(self.goal_library[0])
                n_full_batches = n_old_runs // self.config.progressive_training.train_batch_size
                self.visual_representation.model.eval()
                with torch.no_grad():
                    for batch_idx in range(n_full_batches):
                        x = torch.stack(all_observations[(batch_idx * self.config.progressive_training.train_batch_size) : ((batch_idx + 1) * self.config.progressive_training.train_batch_size)], dim=0)
                        output_goals = self.visual_representation.model.calc_embedding(x, node_path=goal_space_node_path)
                        for idx in range(self.config.progressive_training.train_batch_size):
                            reach_goal_updated = output_goals[idx].squeeze(0).cpu().numpy()
                            self.goal_library[goal_space_idx][(batch_idx * self.config.progressive_training.train_batch_size) + idx] = reach_goal_updated
                            self.goal_library_extent[goal_space_idx][:,0] = np.fmin(self.goal_library_extent[goal_space_idx][:,0], reach_goal_updated)
                            self.goal_library_extent[goal_space_idx][:,1] = np.fmax(self.goal_library_extent[goal_space_idx][:,1], reach_goal_updated)
                    # last batch with remaining indexes:
                    n_remaining_idx = n_old_runs - n_full_batches * self.config.progressive_training.train_batch_size
                    if n_remaining_idx > 0:
                        x = torch.stack(all_observations[(n_full_batches * self.config.progressive_training.train_batch_size) :], dim=0)
                        output_goals = self.visual_representation.model.calc_embedding(x, node_path=goal_space_node_path)
                        for idx in range(n_remaining_idx):
                            reach_goal_updated = output_goals[idx].squeeze(0).cpu().numpy()
                            self.goal_library[goal_space_idx][(n_full_batches * self.config.progressive_training.train_batch_size) + idx] = reach_goal_updated
                            self.goal_library_extent[goal_space_idx][:,0] = np.fmin(self.goal_library_extent[goal_space_idx][:,0], reach_goal_updated)
                            self.goal_library_extent[goal_space_idx][:,1] = np.fmax(self.goal_library_extent[goal_space_idx][:,1], reach_goal_updated) 
                    
                goal_space_idx +=1
                
            ''' Update stage counter'''
            self.counters.stage_idx += 1
            self.counters.last_run_idx_seen_by_nn = run_idx 
            self.counters.n_train_dataset_curr_stage = 0
            self.counters.n_valid_dataset_curr_stage = 0
            
            ''' Save explorer '''
            self.save()
        
        return
