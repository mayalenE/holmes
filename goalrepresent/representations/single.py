import os

from tensorboardX import SummaryWriter

import goalrepresent as gr
from goalrepresent import models


class SingleModelRepresentation(gr.BaseRepresentation):
    """
    Representation with single model Class
    """ 
    @staticmethod
    def default_config():
        default_config = gr.BaseRepresentation.default_config()
        
        # model parameters
        default_config.model = gr.Config()
        default_config.model.name = "VAE"
        default_config.model.config = models.VAEModel.default_config()
        
        # training parameters
        default_config.training = gr.Config()
        default_config.training.output_folder = None
        default_config.training.training_data_type = "iid"  # either "iid" or "sequential"
        default_config.training.n_epochs = 0
        
        default_config.testing = gr.Config()
        default_config.testing.output_folder = None
        default_config.testing.evaluationmodels = gr.Config()
        return default_config
    
    
    def __init__(self, config=None, **kwargs):
        gr.BaseRepresentation.__init__(self, config=config, **kwargs)
        
         # model
        self.set_model(self.config.model.name, self.config.model.config)


    def set_model(self, model_name, model_config):
        model_class = gr.BaseModel.get_model(model_name)
        self.model = model_class(config=model_config)
        
        # update config
        self.config.model.name = model_name
        self.config.model.config = gr.config.update_config(model_config, self.config.model.config)
        
    def run_training(self, train_loader, valid_loader, training_config, keep_best_model=False, logging=True):
        training_config = gr.config.update_config(training_config, self.config.training)

        # prepare output folders
        output_folder = training_config.output_folder
        if (output_folder is not None) and (not os.path.exists (output_folder)):
            os.makedirs(output_folder)
            
        checkpoint_folder = os.path.join(output_folder, "checkpoints")
        if (checkpoint_folder is not None) and (not os.path.exists (checkpoint_folder)):
            os.makedirs(checkpoint_folder)
        self.model.config.checkpoint.folder = checkpoint_folder

        evaluation_folder = os.path.join(output_folder, "evaluation")
        if (evaluation_folder is not None) and (not os.path.exists(evaluation_folder)):
            os.makedirs(evaluation_folder)
        self.model.config.evaluation.folder = evaluation_folder
        
        # prepare logger
        if logging: 
            logging_folder = os.path.join(output_folder, "logging")
            if (logging_folder is not None) and (not os.path.exists (logging_folder)):
                os.makedirs(logging_folder)
    
            logger = SummaryWriter(logging_folder, 'w')
        else:
            logger = None

        # run training
        if training_config.training_data_type == "iid":
            self.model.run_training(train_loader, training_config, valid_loader, logger=logger)
        elif training_config.training_data_type == "sequential":
            self.model.run_sequential_training(train_loader, training_config, valid_loader, logger=logger)
        else:
            raise ValueError('The training data type must be "iid" or "sequential"')
        
        # export scalar data to JSON for external processing
        if logger is not None:
            logger.export_scalars_to_json(os.path.join(output_folder, "output_scalars.json"))
            logger.close()
            
        # if we want the representation to keep the model that performed best on the valid dataset
        if keep_best_model:
            best_model_path = os.path.join(checkpoint_folder, "best_weight_model.pth")
            if os.path.exists(best_model_path):
                best_model = gr.dnn.BaseDNN.load_checkpoint(best_model_path, use_gpu = self.model.config.device.use_gpu)
                cur_n_epochs = self.model.n_epochs
                self.model = best_model
                self.model.n_epochs = cur_n_epochs
        
        # update config
        self.config.training = gr.config.update_config(training_config, self.config.training)


    def run_testing(self, test_loader, testing_config, train_loader=None, valid_loader=None, logging=True):
        # prepare output folders
        output_folder = testing_config.output_folder
        if (output_folder is not None) and (not os.path.exists (output_folder)):
            os.makedirs(output_folder)
            
        # loop over different tests
        test_data = {}
        for k, evalmodel_config in testing_config.evaluationmodels.items():
            output_name, evalmodel_test_data = self.run_evalmodel_testing(test_loader, evalmodel_config, 
                                                                   train_loader=train_loader, valid_loader=valid_loader, 
                                                                   logging=logging)
            
            test_data[output_name] = evalmodel_test_data
            
        
        return test_data
    
    
    def run_evalmodel_testing(self, test_loader, evalmodel_config, train_loader=None, valid_loader=None, logging=True):
        evalmodel_name = evalmodel_config.name
        evalmodel_class = gr.BaseEvaluationModel.get_evaluationmodel(evalmodel_name)
        evalmodel = evalmodel_class(self.model, config=evalmodel_config.config)
        
        # prepare output folders
        curr_output_folder = evalmodel_config.output_folder
        if (curr_output_folder is not None) and (not os.path.exists (curr_output_folder)):
            os.makedirs(curr_output_folder)
        
        checkpoint_folder = os.path.join(curr_output_folder, "checkpoints")
        if (checkpoint_folder is not None) and (not os.path.exists (checkpoint_folder)):
            os.makedirs(checkpoint_folder)
        evalmodel.config.checkpoint.folder = checkpoint_folder
        
        # prepare logger
        if logging: 
            logging_folder = os.path.join(curr_output_folder, "logging")
            if (logging_folder is not None) and (not os.path.exists (logging_folder)):
                os.makedirs(logging_folder)
    
            logger = SummaryWriter(logging_folder, 'w')
        else:
            logger = None
            
        # train the evaluationmodel if needed
        evalmodel.run_training(train_loader=train_loader, valid_loader=valid_loader, logger=logger)
        # test the representation by this evaluationmodel
        evalmodel_test_data = evalmodel.run_representation_testing(test_loader, testing_config=evalmodel_config)
        output_name = curr_output_folder.split("/")[-1]
        
        # export scalar data to JSON for external processing
        if logger is not None:
            logger.export_scalars_to_json(os.path.join(curr_output_folder, "output_scalars.json"))
            logger.close()
            
        # update config
        try:
            old_evalmodel_config = self.config.testing.evaluationmodels[output_name]
        except:
            old_evalmodel_config = None
            
        self.config.testing.evaluationmodels[output_name] = gr.config.update_config(evalmodel_config,old_evalmodel_config)
        
        return output_name, evalmodel_test_data
        


    
