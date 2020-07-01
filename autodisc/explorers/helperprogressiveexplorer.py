import numpy as np

import autodisc as ad


class GSSelectionAlgoFixedProbabilities():

    def __init__(self, rng=None):
        self.rng = rng

    def do_selection(self, probabilities, rng=None):

        if rng is None:
            rng = self.rng if self.rng is not None else np.random.RandomState()

        choice = np.where(np.cumsum(probabilities) > rng.rand())[0][0]

        return choice, probabilities


class GSSelectionMeasureFixedProbabilities():

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def get_measure(self):
        return self.probabilities

    def update_measure(self, target_goal_space_idx, target_goal, reached_goal_per_space, goal_library_extent=None):
        pass


class GSSelectionMeasureScorePerGoalspace():

    def __init__(self, scores):
        self.scores = scores

    def get_measure(self):
        return self.scores

    def update_measure(self, target_goal_space_idx, target_goal, reached_goal_per_space, goal_library_extent=None):
        for reached_goal_idx in range(len(reached_goal_per_space)):
            reached_goal = reached_goal_per_space[reached_goal_idx]
            if np.isnan(reached_goal).sum() == 0:
                self.scores[reached_goal_idx] += 1


class GSSelectionMeasureDiversityPerGoalspace():
    '''
    Measures the average increase of diversity for each goal space over the last n_steps.
    '''

    def __init__(self, n_goal_spaces, config):

        self.n_steps = config.n_steps

        if config.diversity.type.lower() == 'NBinDiversityNBinPerDim'.lower():
            n_bins_per_dimension = config.diversity.n_bins_per_dimension if 'n_bins_per_dimension' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None
            ignore_out_of_range_values = config.diversity.ignore_out_of_range_values if 'ignore_out_of_range_values' in config.diversity else None

            div_class = ad.helper.statistics.NBinDiversityNBinPerDim
            init_params = dict(n_bins_per_dimension=n_bins_per_dimension, fixed_borders=fixed_borders, ignore_out_of_range_values=ignore_out_of_range_values)

        elif config.diversity.type.lower() == 'NBinDiversityNBins'.lower():
            n_bins = config.diversity.n_bins if 'n_bins' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None

            div_class = ad.helper.statistics.NBinDiversityNBins
            init_params = dict(n_bins=n_bins, fixed_borders=fixed_borders)

        else:
            raise ValueError('Unknown diversity type (\'{}\') for goal space selection method!'.format(config.diversity.type))


        self.diversity_measure_per_goalspace = []
        for goalspace_idx in range(n_goal_spaces):
            div_measure = div_class(**init_params)
            self.diversity_measure_per_goalspace.append(div_measure)


    def get_measure(self):

        measure = np.full(len(self.diversity_measure_per_goalspace), np.nan)

        for goalspace_idx, div_measure in enumerate(self.diversity_measure_per_goalspace):

            # add initial diversity of 0 to the history of diversities
            diversity_hist = [0] + div_measure.diversity_per_added_point

            steps = min(self.n_steps, len(diversity_hist)-1)

            m_t = np.array(diversity_hist[-steps:])
            m_t_minus_1 = np.array(diversity_hist[-steps-1:-1])

            diff = m_t - m_t_minus_1

            cur_measure = np.nanmean(diff) if diff.size > 0 else 0

            measure[goalspace_idx] = cur_measure

        return measure


    def update_measure(self, target_goal_space_idx,  target_goal, reached_goal_per_space, goal_library_extent=None):

        # calculate the diversity per goal space
        for goalspace_idx in range(len(reached_goal_per_space)):
            self.diversity_measure_per_goalspace[goalspace_idx].add_point(reached_goal_per_space[goalspace_idx])



class GSSelectionMeasureMeanDiversityOverGoalspaces():
    '''
    Measures the average increase of diversity over all goal spaces dependent on the active goal space over the last n_steps.
    '''


    def __init__(self, n_goal_spaces, config):

        self.n_steps = config.n_steps

        if config.diversity.type.lower() == 'NBinDiversityNBinPerDim'.lower():
            n_bins_per_dimension = config.diversity.n_bins_per_dimension if 'n_bins_per_dimension' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None
            ignore_out_of_range_values = config.diversity.ignore_out_of_range_values if 'ignore_out_of_range_values' in config.diversity else None

            div_class = ad.helper.statistics.NBinDiversityNBinPerDim
            init_params = dict(n_bins_per_dimension=n_bins_per_dimension, fixed_borders=fixed_borders, ignore_out_of_range_values=ignore_out_of_range_values)

        elif config.diversity.type.lower() == 'NBinDiversityNBins'.lower():
            n_bins = config.diversity.n_bins if 'n_bins' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None

            div_class = ad.helper.statistics.NBinDiversityNBins
            init_params = dict(n_bins=n_bins, fixed_borders=fixed_borders)

        else:
            raise ValueError('Unknown diversity type (\'{}\') for goal space selection method!'.format(config.diversity.type))

        self.diversity_measure_per_goalspace = []
        self.active_goalspace_idxs = []

        for goalspace_idx in range(n_goal_spaces):
            div_measure = div_class(**init_params)
            self.diversity_measure_per_goalspace.append(div_measure)


    def get_measure(self):

        measure = np.zeros(len(self.diversity_measure_per_goalspace))

        diff_in_diversity_per_goalspace = []

        for goalspace_idx, div_measure in enumerate(self.diversity_measure_per_goalspace):
            # get the diversity increase of this goal space

            # add initial diversity of 0 to the history of diversities
            diversity_hist = [0] + div_measure.diversity_per_added_point

            m_t = np.array(diversity_hist[1:])
            m_t_minus_1 = np.array(diversity_hist[0:-1])
            diff_in_diversity_per_goalspace.append(m_t - m_t_minus_1)

        # calculate mean over the diff of both goal_spaces
        mean_diff_in_diversity_over_goalspaces = np.nanmean(diff_in_diversity_per_goalspace, 0) if diff_in_diversity_per_goalspace[0].size > 0 else np.array([])

        # change to numpy array to allow boolean indexing
        active_goalspace_idxs = np.array(self.active_goalspace_idxs)

        # get the diff per active goal space
        for goalspace_idx in range(len(self.diversity_measure_per_goalspace)):

            cur_goalspace_inds = (active_goalspace_idxs == goalspace_idx)

            cur_gs_influence_history = mean_diff_in_diversity_over_goalspaces[cur_goalspace_inds]

            cur_measure = np.nanmean(cur_gs_influence_history[-self.n_steps:])

            measure[goalspace_idx] = cur_measure if not np.isnan(cur_measure) else 0

        return measure


    def update_measure(self, target_goal_space_idx, target_goal, reached_goal_per_space, goal_library_extent=None):

        self.active_goalspace_idxs.append(target_goal_space_idx)

        # calculate the diversity per goal space
        for goalspace_idx, reached_goal in enumerate(reached_goal_per_space):
            self.diversity_measure_per_goalspace[goalspace_idx].add_point(reached_goal)



class GSSelectionMeasureDiversityOverCombinedGoalspaces():

    def __init__(self, n_goal_spaces, config):
        self.n_steps = config.n_steps

        self.n_goalspaces = n_goal_spaces

        self.active_goalspace_idxs = []

        if config.diversity.type.lower() == 'NBinDiversityNBinPerDim'.lower():
            n_bins_per_dimension = config.diversity.n_bins_per_dimension if 'n_bins_per_dimension' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None
            ignore_out_of_range_values = config.diversity.ignore_out_of_range_values if 'ignore_out_of_range_values' in config.diversity else None

            self.diversity_measure = ad.helper.statistics.NBinDiversityNBinPerDim(n_bins_per_dimension=n_bins_per_dimension, fixed_borders=fixed_borders, ignore_out_of_range_values=ignore_out_of_range_values)

        elif config.diversity.type.lower() == 'NBinDiversityNBins'.lower():
            n_bins = config.diversity.n_bins if 'n_bins' in config.diversity else None
            fixed_borders = config.diversity.fixed_borders if 'fixed_borders' in config.diversity else None

            self.diversity_measure = ad.helper.statistics.NBinDiversityNBins(n_bins=n_bins, fixed_borders=fixed_borders)

        else:
            raise ValueError('Unknown diversity type (\'{}\') for goal space selection method!'.format(config.diversity.type))


    def get_measure(self):

        measure = np.zeros(self.n_goalspaces)

        # add initial diversity of 0 to the history of diversities
        diversity_hist = [0] + self.diversity_measure.diversity_per_added_point

        # get the diversity increase of this goal space
        m_t = np.array(diversity_hist[1:])
        m_t_minus_1 = np.array(diversity_hist[0:-1])
        diff_in_diversity = m_t - m_t_minus_1

        # change to numpy array to allow boolean indexing
        active_goalspace_idxs = np.array(self.active_goalspace_idxs)

        # get the diff per active goal space
        for goalspace_idx in range(self.n_goalspaces):

            cur_goalspace_inds = (active_goalspace_idxs == goalspace_idx)

            cur_gs_influence_history = diff_in_diversity[cur_goalspace_inds]

            measure[goalspace_idx] = np.nanmean(cur_gs_influence_history[-self.n_steps:]) if cur_gs_influence_history.size > 0 else 0

        return measure


    def update_measure(self, target_goal_space_idx, target_goal, reached_goal_per_space, goal_library_extent=None):

        self.active_goalspace_idxs.append(target_goal_space_idx)

        # combine all goal spaces:
        combined_reached_point = np.hstack(tuple(reached_goal for reached_goal in reached_goal_per_space))
        self.diversity_measure.add_point(combined_reached_point)


class GSSelectionMeasureCompetencePerGoalSpace():

    def __init__(self, explorer, n_goal_spaces, config):
        self.explorer = explorer
        self.n_steps = config.n_steps
        self.unnormalized_competence_per_goalspace = [[] for _ in range(n_goal_spaces)]
        self.max_distance_per_goalspace = np.zeros(n_goal_spaces)


    def get_measure(self):

        measure = np.zeros(len(self.unnormalized_competence_per_goalspace))

         # get the diff per active goal space
        for goalspace_idx, competence_hist in enumerate(self.unnormalized_competence_per_goalspace):

            cur_measure = np.nanmean(competence_hist[-self.n_steps:]) if len(competence_hist) > 0 else 0

            # normalize competence distance
            measure[goalspace_idx] = cur_measure / self.max_distance_per_goalspace[goalspace_idx] if self.max_distance_per_goalspace[goalspace_idx] > 0 else cur_measure

        return measure


    def update_measure(self, target_goal_space_idx, target_goal, reached_goal_per_space, goal_library_extent=None):

        n_goalspaces = self.max_distance_per_goalspace.shape[0]

        # only compute based on possible source policies
        possible_run_inds_per_goalspace = [self.explorer.get_possible_source_policies_inds(target_goal_space_idx) for target_goal_space_idx in range(n_goalspaces)]

        # update max distances
        for goal_space_idx in range(n_goalspaces):
            if np.sum(possible_run_inds_per_goalspace[goal_space_idx]) > 0:
                new_distance = self.explorer.goal_space_representations[goal_space_idx].calc_distance(reached_goal_per_space[goal_space_idx], self.explorer.goal_library[goal_space_idx][possible_run_inds_per_goalspace[goal_space_idx], :], goal_library_extent[goal_space_idx])
                max_distance = np.nanmax(new_distance)

                self.max_distance_per_goalspace[goal_space_idx] = max(max_distance, self.max_distance_per_goalspace[goal_space_idx])



        if not np.isnan(target_goal_space_idx):

            shortest_optimal_distance = 0

            if np.sum(possible_run_inds_per_goalspace[goal_space_idx]) > 0:
                # minimal distance between current goal and any already reached point in goal space
                goal_distances = self.explorer.goal_space_representations[target_goal_space_idx].calc_distance(target_goal, self.explorer.goal_library[target_goal_space_idx][possible_run_inds_per_goalspace[goal_space_idx], :], goal_library_extent[target_goal_space_idx])
                shortest_optimal_distance = np.nanmin(goal_distances)

            # distance current goal and current reached point in goal space
            current_distance = self.explorer.goal_space_representations[target_goal_space_idx].calc_distance(target_goal, reached_goal_per_space[target_goal_space_idx], goal_library_extent[goal_space_idx])
            current_distance = current_distance

            competence = shortest_optimal_distance - current_distance

            self.unnormalized_competence_per_goalspace[target_goal_space_idx].append(competence)