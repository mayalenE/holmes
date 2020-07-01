import autodisc as ad
import goalrepresent as gr

def get_system_config():
    system_config = ad.systems.Lenia.default_config()
    system_config.version = "pytorch_fft"
    system_config.use_gpu = True
    return system_config

def get_system_parameters():
    system_parameters = ad.systems.Lenia.default_system_parameters()
    system_parameters.size_y = 256
    system_parameters.size_x = 256

    return system_parameters

def get_model_config(model_name):
    model_class = eval("gr.models.{}Model".format(model_name))
    model_config = model_class.default_config()

    if 'network' in model_config:
    
        if "ProgressiveTree" in model_name:
            ## network

            model_config.node_classname = "VAE"
            model_config.node.network.name = "Burgess"
            model_config.node.network.parameters = {"n_channels": 1, "input_size": (256,256), "n_latents": 16, "n_conv_layers": 6, "hidden_channels": 16, "hidden_dim": 64, "encoder_conditional_type": "gaussian", "feature_layer": 2}
            model_config.node.create_connections = {"lf": True,"gf": False, "gfi":True, "lfi": True, "recon": True, }

            model_config.network.parameters = {"n_channels": 1, "input_size": (256,256), "n_latents": 16, "n_conv_layers": 6, "hidden_channels": 16, "hidden_dim": 64, "encoder_conditional_type": "gaussian", "feature_layer": 2}


            ## device
            model_config.node.device.use_gpu = True

        else:
            ## network
            model_config.network.name = "Burgess"
            model_config.network.parameters = {"n_channels": 1, "input_size": (256,256), "n_latents": 16, "n_conv_layers": 6, "hidden_channels": 16, "hidden_dim": 64, "encoder_conditional_type": "gaussian", "feature_layer": 2}

        ## initialization
        model_config.network.initialization.name = "kaiming_uniform"
        model_config.network.initialization.parameters =  {}

        ## loss
        model_config.loss.name = "VAE"
        model_config.loss.parameters = {"reconstruction_dist": "bernoulli"}

        ## optimizer
        model_config.optimizer.name = "Adam"
        model_config.optimizer.parameters = {"lr": 1e-3, "weight_decay": 1e-5 }

         # device
        model_config.device.use_gpu = True

        ## logging
        model_config.logging.record_valid_images_every = 100
        model_config.logging.record_embeddings_every = 400

        ## checkpoint
        model_config.checkpoint.save_model_every = 1

        ## evaluation
        model_config.evaluation.save_results_every = 5000
    
    return model_config

def get_explorer_config():

    explorer_config = ad.explorers.ProgressiveExplorer.default_config()
    explorer_config.seed = 8
    explorer_config.num_of_random_initialization = 1000

    explorer_config.run_parameters = []

    # Parameter 1: init state
    parameter = ad.Config()
    parameter.name = 'init_state'
    parameter.type = 'cppn_evolution'

    parameter.init = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.init.neat_config_file = 'neat_config.cfg'
    parameter.init.n_generations = 1
    parameter.init.best_genome_of_last_generation = True

    parameter.mutate = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.mutate.neat_config_file = 'neat_config.cfg'
    parameter.mutate.n_generations = 2
    parameter.mutate.best_genome_of_last_generation = True

    explorer_config.run_parameters.append(parameter)

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    explorer_config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    explorer_config.run_parameters.append(parameter)

    # Parameter 4: b
    parameter = ad.Config()
    parameter.name = 'b'
    parameter.type = 'sampling'
    parameter.init = ('function', ad.helper.sampling.sample_vector, (('discrete', 1, 3), (0, 1)))
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': 0.1, 'min': 0, 'max': 1}
    explorer_config.run_parameters.append(parameter)

    # Parameter 5: m
    parameter = ad.Config()
    parameter.name = 'm'
    parameter.type = 'sampling'
    parameter.init = ('continuous', 0, 1)
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': 0.1, 'min': 0, 'max': 1}
    explorer_config.run_parameters.append(parameter)

    # Parameter 6: s
    parameter = ad.Config()
    parameter.name = 's'
    parameter.type = 'sampling'
    parameter.init = ('continuous', 0.001, 0.3)
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': 0.05, 'min': 0.001, 'max': 0.3}
    explorer_config.run_parameters.append(parameter)
        
    # visual representation
    explorer_config.visual_representation = gr.representations.SingleModelRepresentation.default_config()
    explorer_config.visual_representation.seed = 8
    explorer_config.visual_representation.training.output_folder = "./training"
    explorer_config.visual_representation.model.name = "ProgressiveTree"
    explorer_config.visual_representation.model.config = get_model_config(explorer_config.visual_representation.model.name)

    
    # goal space selection
    explorer_config.goal_space_selection.type = 'adaptive'
    if explorer_config.goal_space_selection.type in ['probability_distribution']:
        explorer_config.goal_space_selection.distribution = None

    elif explorer_config.goal_space_selection.type in ['adaptive']:
        explorer_config.goal_space_selection.measure = ad.Config()
        explorer_config.goal_space_selection.measure.type = 'score_per_goalspace'
        explorer_config.goal_space_selection.measure.n_steps = None

        if None is not None and None is not None:
            raise ValueError('Only explorer_config.goal_space_selection.measure.n_bins_per_dimension or explorer_config.goal_space_selection.measure.n_bins can be defined!')
        if None is not None:
            explorer_config.goal_space_selection.measure.diversity = ad.Config()
            explorer_config.goal_space_selection.measure.diversity.type = 'NBinDiversityNBinPerDim'
            explorer_config.goal_space_selection.measure.diversity.n_bins_per_dimension = None
        elif None is not None:
            explorer_config.goal_space_selection.measure.diversity = ad.Config()
            explorer_config.goal_space_selection.measure.diversity.type = 'NBinDiversityNBins'
            explorer_config.goal_space_selection.measure.diversity.n_bins = None
            
        # add constraint to the diversity measure
        explorer_config.goal_space_selection.measure.update_constraints = dict( active = 'after_split', filter = (('statistics.is_dead', '==', False), 'and', ('statistics.classifier_animal', '==', False)))

        explorer_config.goal_space_selection.selection_algo = ad.Config()
        explorer_config.goal_space_selection.selection_algo.type = 'softmax'
        if explorer_config.goal_space_selection.selection_algo.type in ['epsilon_greedy']:
            explorer_config.goal_space_selection.selection_algo.epsilon = None
        elif explorer_config.goal_space_selection.selection_algo.type in ['softmax']:
            explorer_config.goal_space_selection.selection_algo.beta = 5
        elif explorer_config.goal_space_selection.selection_algo.type in ['epsilon_softmax']:
            explorer_config.goal_space_selection.selection_algo.epsilon = None
            explorer_config.goal_space_selection.selection_algo.beta = 5
            
    # goal selection
    explorer_config.goal_selection.type = 'random'
    explorer_config.goal_selection.sampling_from_reached_boundaries = ad.Config()
    explorer_config.goal_selection.sampling_from_reached_boundaries.margin_min = 0
    explorer_config.goal_selection.sampling_from_reached_boundaries.margin_max = 0
    
    # progressive growing parameters
    explorer_config.progressive_growing.split_trigger = ad.Config(dict(active = True, fitness_key= 'recon', type= 'plateau', parameters= dict(epsilon= 20, n_steps_average= 50), conditions= dict(min_init_n_epochs = 2000, n_min_points = 500, n_max_splits= 10, n_epochs_min_between_splits= 200), save_model_before_after= True))
    explorer_config.progressive_growing.split_trigger.boundary_config = {"z_fitness": "recon_loss", "algo": "cluster.KMeans"}
    
    # progressive training parameters
    explorer_config.progressive_training.dataset_constraints = [dict( active = True, filter = ('statistics.is_dead', '==', False))]
    explorer_config.progressive_training.dataset_augment = True
    explorer_config.progressive_training.n_runs_between_stages = 100
    explorer_config.progressive_training.n_epochs_per_stage = 100
    explorer_config.progressive_training.train_batch_size = 128
    explorer_config.progressive_training.importance_sampling_new_vs_old = 0.3
    explorer_config.progressive_training.alternated_backward = {"active": True, "ratio_epochs": {"connections": 2, "core": 8}}


    # how are the source policies for a mutation are selected
    explorer_config.source_policy_selection.type = 'optimal'
    explorer_config.source_policy_selection.constraints = []


    return explorer_config

def get_number_of_explorations():
    return 5000
