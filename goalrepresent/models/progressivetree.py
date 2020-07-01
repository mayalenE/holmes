from copy import deepcopy
import goalrepresent as gr
from goalrepresent import dnn, models
from goalrepresent.dnn.solvers import initialization
from goalrepresent.helper import tensorboardhelper
from goalrepresent.helper.misc import do_filter_boolean
from goalrepresent.datasets.image.imagedataset import MIXEDDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import warnings


# from torchviz import make_dot

class Node(nn.Module):
    """
    Node base class
    """

    def __init__(self, depth, **kwargs):
        self.depth = depth
        self.leaf = True  # set to Fale when node is split
        self.boundary = None
        self.feature_range = None
        self.leaf_accumulator = []
        self.fitness_last_epochs = []

    def reset_accumulator(self):
        self.leaf_accumulator = []
        if not self.leaf:
            self.left.reset_accumulator()
            self.right.reset_accumulator()

    def get_child_node(self, path):
        node = self
        for d in range(1, len(path)):
            if path[d] == "0":
                node = node.left
            else:
                node = node.right

        return node

    def get_leaf_pathes(self, path_taken=None):
        if path_taken is None:
            path_taken = []
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])

        else:
            left_leaf_accumulator = self.left.get_leaf_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_leaf_pathes(path_taken=path_taken + "1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def get_node_pathes(self, path_taken=None):
        if path_taken is None:
            path_taken = []
        if self.depth == 0:
            path_taken = "0"
        if self.leaf:
            return ([path_taken])

        else:
            self.leaf_accumulator.extend([path_taken])
            left_leaf_accumulator = self.left.get_node_pathes(path_taken=path_taken + "0")
            self.leaf_accumulator.extend(left_leaf_accumulator)
            right_leaf_accumulator = self.right.get_node_pathes(path_taken=path_taken + "1")
            self.leaf_accumulator.extend(right_leaf_accumulator)
            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def create_boundary(self, z_library, z_fitness=None, boundary_config=None):
        # normalize z points
        self.feature_range = (z_library.min(axis=0), z_library.max(axis=0))
        X = z_library - self.feature_range[0]
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        X = X / scale

        default_boundary_config = gr.Config()
        default_boundary_config.algo = 'svm.SVC'
        default_boundary_config.kwargs = gr.Config()
        if boundary_config is not None:
            boundary_config = gr.config.update_config(boundary_config, default_boundary_config)
        boundary_algo = eval(boundary_config.algo)

        if z_fitness is None:
            if boundary_config.algo == 'cluster.KMeans':
                boundary_config.kwargs.n_clusters = 2
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X)
        else:
            y = z_fitness.squeeze()
            if boundary_config.algo == 'cluster.KMeans':
                center0 = np.median(X[y <= np.percentile(y, 20), :], axis=0)
                center1 = np.median(X[y > np.percentile(y, 80), :], axis=0)
                center = np.stack([center0, center1])
                boundary_config.kwargs.init = center
                boundary_config.kwargs.n_clusters = 2
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X)
            elif boundary_config.algo == 'svm.SVC':
                y = y > np.percentile(y, 80)
                self.boundary = boundary_algo(**boundary_config.kwargs).fit(X, y)

        return

    def depth_first_forward(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"
            x_ids = list(range(len(x)))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def depth_first_forward_whole_branch_preorder(self, x, tree_path_taken=None, x_ids=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"
            x_ids = list(range(len(x)))

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, x_ids, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, x_ids, node_outputs]])

            z = node_outputs["z"]
            x_side = self.get_children_node(z)
            x_ids_left = np.where(x_side == 0)[0]
            if len(x_ids_left) > 0:
                left_leaf_accumulator = self.left.depth_first_forward_whole_branch_preorder(x[x_ids_left], [path + "0" for path in
                                                                                      [tree_path_taken[x_idx] for x_idx
                                                                                       in x_ids_left]],
                                                                      [x_ids[x_idx] for x_idx in x_ids_left],
                                                                      parent_lf[x_ids_left],
                                                                      parent_gf[x_ids_left],
                                                                      parent_gfi[x_ids_left],
                                                                      parent_lfi[x_ids_left],
                                                                      parent_recon_x[x_ids_left])
                self.leaf_accumulator.extend(left_leaf_accumulator)

            x_ids_right = np.where(x_side == 1)[0]
            if len(x_ids_right) > 0:
                right_leaf_accumulator = self.right.depth_first_forward_whole_branch_preorder(x[x_ids_right], [path + "1" for path in
                                                                                         [tree_path_taken[x_idx] for
                                                                                          x_idx in x_ids_right]],
                                                                        [x_ids[x_idx] for x_idx in x_ids_right],
                                                                        parent_lf[x_ids_right],
                                                                        parent_gf[x_ids_right],
                                                                        parent_gfi[x_ids_right],
                                                                        parent_lfi[x_ids_right],
                                                                        parent_recon_x[x_ids_right])
                self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def depth_first_forward_whole_tree_preorder(self, x, tree_path_taken=None, parent_lf=None, parent_gf=None,
                            parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if self.depth == 0:
            tree_path_taken = ["0"] * len(x)  # all the paths start with "O"

        node_outputs = self.node_forward(x, parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)

        parent_lf = node_outputs["lf"].detach()
        parent_gf = node_outputs["gf"].detach()
        parent_gfi = node_outputs["gfi"].detach()
        parent_lfi = node_outputs["lfi"].detach()
        parent_recon_x = node_outputs["recon_x"].detach()

        if self.leaf:
            # return path_taken, x_ids, leaf_outputs
            return ([[tree_path_taken, node_outputs]])

        else:
            self.leaf_accumulator.extend([[tree_path_taken, node_outputs]])
            #send everything left
            left_leaf_accumulator = self.left.depth_first_forward_whole_tree_preorder(x, [path+"0" for path in tree_path_taken],
                                                                  parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
            self.leaf_accumulator.extend(left_leaf_accumulator)

            #send everything right
            right_leaf_accumulator = self.right.depth_first_forward_whole_tree_preorder(x, [path+"1" for path in tree_path_taken],
                                                                    parent_lf, parent_gf, parent_gfi, parent_lfi, parent_recon_x)
            self.leaf_accumulator.extend(right_leaf_accumulator)

            leaf_accumulator = self.leaf_accumulator
            self.reset_accumulator()

        return leaf_accumulator

    def node_forward(self, x, parent_lf=None, parent_gf=None, parent_gfi=None, parent_lfi=None,
                     parent_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x, parent_lf, parent_gf, parent_gfi, parent_lfi,
                                                  parent_recon_x)
        if self.depth == 0:
            encoder_outputs = self.network.encoder(x)
        else:
            encoder_outputs = self.network.encoder(x, parent_lf, parent_gf)
        model_outputs = self.node_forward_from_encoder(encoder_outputs, parent_gfi, parent_lfi, parent_recon_x)
        return model_outputs

    def get_boundary_side(self, z):
        if self.boundary is None:
            raise ValueError("Boundary computation is required before calling this function")
        else:
            # normalize
            if isinstance(z, torch.Tensor):
                z = z.detach().cpu().numpy()
            z = z - self.feature_range[0]
            scale = self.feature_range[1] - self.feature_range[0]
            scale[np.where(
                scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
            z = z / scale
            # compute boundary side
            side = self.boundary.predict(z)  # returns 0: left, 1: right
        return side

    def get_children_node(self, z):
        """
        Return code: -1 for leaf node,  0 to send left, 1 to send right
        """
        if self.leaf:
            return -1 * torch.ones_like(z)
        else:
            return self.get_boundary_side(z)

    def generate_images(self, n_images, from_node_path='', from_side = None, parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        self.eval()
        with torch.no_grad():

            if len(from_node_path) == 0:
                if from_side is None or self.boundary is None:
                    z_gen = torch.randn((n_images, self.config.network.parameters.n_latents))
                else:
                    desired_side = int(from_side)
                    z_gen = []
                    remaining = n_images
                    max_trials = n_images
                    trials = 0
                    while remaining > 0:
                        cur_z_gen = torch.randn((remaining, self.config.network.parameters.n_latents))
                        if trials == max_trials:
                            z_gen.append(cur_z_gen)
                            remaining = 0
                            break;
                        cur_z_gen_side = self.get_children_node(cur_z_gen)
                        if desired_side == 0:
                            left_side_ids = np.where(cur_z_gen_side == 0)[0]
                            if len(left_side_ids) > 0:
                                z_gen.append(cur_z_gen[left_side_ids[:remaining]])
                                remaining -= len(left_side_ids)
                        elif desired_side == 1:
                            right_side_ids = np.where(cur_z_gen_side == 1)[0]
                            if len(right_side_ids) > 0:
                                z_gen.append(cur_z_gen[right_side_ids[:remaining]])
                                remaining -= len(right_side_ids)
                        else:
                            raise ValueError("wrong path")
                        trials += 1
                    z_gen = torch.cat(z_gen)

            else:
                desired_side = int(from_node_path[0])
                z_gen = []
                remaining = n_images
                max_trials = n_images
                trials = 0
                while remaining > 0:
                    cur_z_gen = torch.randn((remaining, self.config.network.parameters.n_latents))
                    if trials == max_trials:
                        z_gen.append(cur_z_gen)
                        remaining = 0
                        break;
                    cur_z_gen_side = self.get_children_node(cur_z_gen)
                    if desired_side == 0:
                        left_side_ids = np.where(cur_z_gen_side == 0)[0]
                        if len(left_side_ids) > 0:
                            z_gen.append(cur_z_gen[left_side_ids[:remaining]])
                            remaining -= len(left_side_ids)
                    elif desired_side == 1:
                        right_side_ids = np.where(cur_z_gen_side == 1)[0]
                        if len(right_side_ids) > 0:
                            z_gen.append(cur_z_gen[right_side_ids[:remaining]])
                            remaining -= len(right_side_ids)
                    else:
                        raise ValueError("wrong path")
                    trials += 1
                z_gen = torch.cat(z_gen)

            z_gen = self.push_variable_to_device(z_gen)
            node_outputs = self.node_forward_from_encoder({'z': z_gen}, parent_gfi, parent_lfi, parent_recon_x)
            gfi = node_outputs["gfi"].detach()
            lfi = node_outputs["lfi"].detach()
            recon_x = node_outputs["recon_x"].detach()

            if len(from_node_path) == 0:
                if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                    recon_x = torch.sigmoid(recon_x)
                recon_x = recon_x.detach()
                return recon_x

            else:
                if desired_side == 0:
                    return self.left.generate_images(n_images, from_node_path=from_node_path[1:], from_side = from_side, parent_gfi=gfi, parent_lfi=lfi, parent_recon_x=recon_x)
                elif desired_side == 1:
                    return self.right.generate_images(n_images, from_node_path=from_node_path[1:], from_side = from_side, parent_gfi=gfi, parent_lfi=lfi, parent_recon_x=recon_x)


def get_node_class(base_class):

    class NodeClass(Node, base_class):
        def __init__(self, depth, parent_network=None, config=None, **kwargs):
            base_class.__init__(self, config=config, **kwargs)
            Node.__init__(self, depth, **kwargs)

            # connect encoder and decoder
            if self.depth > 0:
                self.network.encoder = ConnectedEncoder(parent_network.encoder, depth, connect_lf=config.create_connections["lf"],
                                                        connect_gf=config.create_connections["gf"])
                self.network.decoder = ConnectedDecoder(parent_network.decoder, depth, connect_gfi=config.create_connections["gfi"],
                                                        connect_lfi=config.create_connections["lfi"],
                                                        connect_recon=config.create_connections["recon"])

            self.set_device(self.config.device.use_gpu)

        def node_forward_from_encoder(self, encoder_outputs, parent_gfi=None, parent_lfi=None,
                                      parent_recon_x=None):
            if self.depth == 0:
                decoder_outputs = self.network.decoder(encoder_outputs["z"])
            else:
                decoder_outputs = self.network.decoder(encoder_outputs["z"], parent_gfi, parent_lfi,
                                                       parent_recon_x)
            model_outputs = encoder_outputs
            model_outputs.update(decoder_outputs)
            return model_outputs

    return NodeClass

# possible node classes:
VAENode_local = get_node_class(models.VAEModel)
VAENode = type('VAENode', (Node, models.VAEModel), dict(VAENode_local.__dict__))

BetaVAENode_local = get_node_class(models.BetaVAEModel)
BetaVAENode = type('BetaVAENode', (Node, models.BetaVAEModel), dict(BetaVAENode_local.__dict__))

AnnealedVAENode_local = get_node_class(models.AnnealedVAEModel)
AnnealedVAENode = type('AnnealedVAENode', (Node, models.AnnealedVAEModel), dict(AnnealedVAENode_local.__dict__))

BetaTCVAENode_local = get_node_class(models.BetaTCVAEModel)
BetaTCVAENode = type('BetaTCVAENode', (Node, models.BetaTCVAEModel), dict(BetaTCVAENode_local.__dict__))


class ProgressiveTreeModel(dnn.BaseDNN, gr.BaseModel):
    """
    dnn with tree network, loss which is based on leaf's losses, optimizer from that loss
    """

    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # tree config
        default_config.tree = gr.Config()

        # node config
        default_config.node_classname = "VAE"
        default_config.node = eval("gr.models.{}Model.default_config()".format(default_config.node_classname))
        default_config.node.create_connections = {"lf": True, "gf": False, "gfi": True, "lfi": True, "recon": True}

        # loss parameters
        default_config.loss.name = "VAE"

        # optimizer
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())
        self.NodeClass = eval("{}Node".format(self.config.node_classname))
        self.split_history = {}  # dictionary with node path keys and boundary values

        dnn.BaseDNN.__init__(self, config=config, **kwargs)

    def set_network(self, network_name, network_parameters):
        depth = 0
        self.network = self.NodeClass(depth, config=self.config.node)  # root node that links to child nodes
        self.network.optimizer_group_id = 0
        self.output_keys_list = self.network.output_keys_list + ["path_taken"]  # node.left is a leaf node

        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        # give only trainable parameters
        trainable_parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(trainable_parameters, **optimizer_parameters)
        self.network.optimizer_group_id = 0
        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def forward_for_graph_tracing(self, x):
        pass

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)
        x = self.push_variable_to_device(x)
        is_train = self.network.training
        if len(x) == 1 and is_train:
            self.network.eval()
            depth_first_traversal_outputs = self.network.depth_first_forward(x)
            self.network.train()
        else:
            depth_first_traversal_outputs = self.network.depth_first_forward(x)

        model_outputs = {}
        x_order_ids = []
        for leaf_idx in range(len(depth_first_traversal_outputs)):
            cur_node_path = depth_first_traversal_outputs[leaf_idx][0]
            cur_node_x_ids = depth_first_traversal_outputs[leaf_idx][1]
            cur_node_outputs = depth_first_traversal_outputs[leaf_idx][2]
            # stack results
            if not model_outputs:
                model_outputs["path_taken"] = cur_node_path
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = v
            else:
                model_outputs["path_taken"] += cur_node_path
                for k, v in cur_node_outputs.items():
                    model_outputs[k] = torch.cat([model_outputs[k], v], dim=0)
            # save the sampled ids to reorder as in the input batch at the end
            x_order_ids += list(cur_node_x_ids)

        # reorder points
        sort_order = tuple(np.argsort(x_order_ids))
        for k, v in model_outputs.items():
            if isinstance(v, list):
                model_outputs[k] = [v[i] for i in sort_order]
            else:
                model_outputs[k] = v[sort_order, :]

        return model_outputs

    def calc_embedding(self, x, node_path=None, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if node_path is None:
            warnings.warn("WARNING: computing the embedding in root node of progressive tree as no path specified")
            node_path = "0"
        n_latents = self.config.network.parameters.n_latents
        z = torch.Tensor().new_full((len(x), n_latents), float("nan"))
        x = self.push_variable_to_device(x)
        self.eval()
        with torch.no_grad():
            all_nodes_outputs = self.network.depth_first_forward_whole_branch_preorder(x)
            for node_idx in range(len(all_nodes_outputs)):
                cur_node_path = all_nodes_outputs[node_idx][0][0]
                if cur_node_path != node_path:
                    continue;
                else:
                    cur_node_x_ids = all_nodes_outputs[node_idx][1]
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    for idx in range(len(cur_node_x_ids)):
                        z[cur_node_x_ids[idx]] = cur_node_outputs["z"][idx]
                    break;
        return z

    def split_node(self, node_path, split_trigger=None, x_loader=None, x_fitness=None):
        """
        z_library: n_samples * n_latents
        z_fitness: n_samples * 1 (eg: reconstruction loss)
        """
        node = self.network.get_child_node(node_path)
        node.NodeClass = type(node)

        # save model
        if split_trigger is not None and (split_trigger.save_model_before_after or split_trigger.save_model_before_after == 'before'):
            self.save_checkpoint(os.path.join(self.config.checkpoint.folder,
                                              'weight_model_before_split_{}_node_{}_epoch_{}.pth'.format(
                                                  len(self.split_history)+1, node_path, self.n_epochs)))

        # (optional) Train for X epoch parent with new data (+ replay optional)
        #TODO: add logger
        if x_loader is not None and split_trigger.n_epochs_before_split > 0:
            for epoch_before_split in range(split_trigger.n_epochs_before_split):
                _ = self.train_epoch(x_loader, logger=None)

        self.eval()
        # Create boundary
        if x_loader is not None:
            with torch.no_grad():
                z_library = self.calc_embedding(x_loader.dataset.images, node_path=node_path).detach().cpu().numpy()
                if split_trigger.boundary_config.z_fitness is None:
                    z_fitness = None
                else:
                    z_fitness = x_fitness
                if np.isnan(z_library).any():
                    keep_ids = ~(np.isnan(z_library.sum(1)))
                    z_library = z_library[keep_ids]
                    if z_fitness is not None:
                        x_fitness = x_fitness[keep_ids]
                        z_fitness = x_fitness
                node.create_boundary(z_library, z_fitness, boundary_config=split_trigger.boundary_config)

        # Instanciate childrens
        node.left = node.NodeClass(node.depth + 1, parent_network=deepcopy(node.network), config=node.config)
        node.right = node.NodeClass(node.depth + 1, parent_network=deepcopy(node.network), config=node.config)
        node.leaf = False

        # Freeze parent parameters
        for param in node.network.parameters():
            param.requires_grad = False

        # Update optimize
        cur_node_optimizer_group_id = node.optimizer_group_id
        del self.optimizer.param_groups[cur_node_optimizer_group_id]
        node.optimizer_group_id = None
        n_groups = len(self.optimizer.param_groups)
        # update optimizer_group ids in the tree and sanity check that there is no conflict
        sanity_check = np.asarray([False] * n_groups)
        for leaf_path in self.network.get_leaf_pathes():
            if leaf_path[:len(node_path)] != node_path:
                other_leaf = self.network.get_child_node(leaf_path)
                if other_leaf.optimizer_group_id > cur_node_optimizer_group_id:
                    other_leaf.optimizer_group_id -= 1
                if not sanity_check[other_leaf.optimizer_group_id]:
                    sanity_check[other_leaf.optimizer_group_id] = True
                else:
                    raise ValueError("doublons in the optimizer group ids")
        if (n_groups > 0) and (~sanity_check).any():
            raise ValueError("optimizer group ids does not match the optimzer param groups length")
        self.optimizer.add_param_group({"params": node.left.parameters()})
        node.left.optimizer_group_id = n_groups
        self.optimizer.add_param_group({"params": node.right.parameters()})
        node.right.optimizer_group_id = n_groups + 1


        # save split history
        self.split_history[node_path] = {"depth": node.depth, "leaf": node.leaf, "boundary": node.boundary,
                                         "feature_range": node.feature_range, "epoch": self.n_epochs}
        if x_loader is not None:
            self.save_split_history(node_path, x_loader, z_library, x_fitness, split_trigger)


        # save model
        if split_trigger is not None and (split_trigger.save_model_before_after or split_trigger.save_model_before_after == 'after'):
            self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'weight_model_after_split_{}_node_{}_epoch_{}.pth'.format(len(self.split_history), node_path, self.n_epochs)))


        return


    def save_split_history(self, node_path, x_loader, z_library, z_fitness, split_trigger):
        # save results
        split_output_folder = os.path.join(self.config.evaluation.folder, "split_history", "split_{}_node_{}_epoch_{}".format(len(self.split_history), node_path, self.n_epochs))
        if not os.path.exists(split_output_folder):
            os.makedirs(split_output_folder)

        ## a) save z_library (z_fitness, y_poor_buffer, y_predicted, y_ground_truth)
        node = self.network.get_child_node(node_path)
        y_predicted = node.get_children_node(z_library)
        y_ground_truth = x_loader.dataset.labels
        np.savez(os.path.join(split_output_folder, "z_library.npz"),
                 **{'z_library': z_library, 'z_fitness': z_fitness,
                    'y_predicted': y_predicted, 'y_ground_truth': y_ground_truth})

        ## b) poor data make grid
        if split_trigger.type == "threshold":
            output_filename = os.path.join(split_output_folder, "poor_data_buffer.png")
            poor_data_buffer = x_loader.dataset.images[np.where(z_fitness > split_trigger.parameters.threshold)[0]]
            img = np.transpose(make_grid(poor_data_buffer, nrow=int(np.sqrt(poor_data_buffer.shape[0])), padding=0).cpu().numpy(), (1, 2, 0))
            plt.figure()
            plt.imshow(img)
            plt.savefig(output_filename)
            plt.close()

        ## c) left/right samples from which boundary is fitted
        if split_trigger.boundary_config.z_fitness == "recon_loss":
            y_fit = z_fitness > np.percentile(z_fitness, 80)
            samples_left_fit_ids = np.where(y_fit == 0)[0]
            samples_left_fit_ids = np.random.choice(samples_left_fit_ids, min(100, len(samples_left_fit_ids)))
            output_filename = os.path.join(split_output_folder, "samples_left_fit.png")
            samples_left_fit_buffer = x_loader.dataset.images[samples_left_fit_ids]
            img = np.transpose(
                make_grid(samples_left_fit_buffer, nrow=int(np.sqrt(samples_left_fit_buffer.shape[0])),
                          padding=0).cpu().numpy(), (1, 2, 0))
            plt.figure()
            plt.imshow(img)
            plt.savefig(output_filename)
            plt.close()

            samples_right_fit_ids = np.where(y_fit == 1)[0]
            samples_right_fit_ids = np.random.choice(samples_right_fit_ids, min(100, len(samples_right_fit_ids)))
            output_filename = os.path.join(split_output_folder, "samples_right_fit.png")
            samples_right_fit_buffer = x_loader.dataset.images[samples_right_fit_ids]
            img = np.transpose(
                make_grid(samples_right_fit_buffer, nrow=int(np.sqrt(samples_right_fit_buffer.shape[0])),
                          padding=0).cpu().numpy(), (1, 2, 0))
            plt.figure()
            plt.imshow(img)
            plt.savefig(output_filename)
            plt.close()

            ## d) wrongly classified buffer make grid
            wrongly_sent_left_ids = np.where((y_predicted == 0) & (y_predicted != y_fit))[0]
            if len(wrongly_sent_left_ids) > 0:
                wrongly_sent_left_ids = np.random.choice(wrongly_sent_left_ids, min(100, len(wrongly_sent_left_ids)))
                output_filename = os.path.join(split_output_folder, "wrongly_sent_left.png")
                wrongly_sent_left_buffer = x_loader.dataset.images[wrongly_sent_left_ids]
                img = np.transpose(
                    make_grid(wrongly_sent_left_buffer, nrow=int(np.sqrt(wrongly_sent_left_buffer.shape[0])),
                              padding=0).cpu().numpy(), (1, 2, 0))
                plt.figure()
                plt.imshow(img)
                plt.savefig(output_filename)
                plt.close()

            wrongly_sent_right_ids = np.where((y_predicted == 1) & (y_predicted != y_fit))[0]
            if len(wrongly_sent_right_ids) > 0:
                wrongly_sent_right_ids = np.random.choice(wrongly_sent_right_ids, min(100, len(wrongly_sent_right_ids)))
                output_filename = os.path.join(split_output_folder, "wrongly_sent_right.png")
                wrongly_sent_right_buffer = x_loader.dataset.images[wrongly_sent_right_ids]
                img = np.transpose(
                    make_grid(wrongly_sent_right_buffer, nrow=int(np.sqrt(wrongly_sent_right_buffer.shape[0])),
                              padding=0).cpu().numpy(), (1, 2, 0))
                plt.figure()
                plt.imshow(img)
                plt.savefig(output_filename)
                plt.close()


        ## d) left and right side generated samples
        output_filename = os.path.join(split_output_folder, "samples_gen_sent_left.png")
        samples_left_gen_buffer = self.network.generate_images(100, from_node_path=node_path[1:], from_side='0')
        img = np.transpose(
            make_grid(samples_left_gen_buffer, nrow=int(np.sqrt(samples_left_gen_buffer.shape[0])),
                      padding=0).cpu().numpy(), (1, 2, 0))
        plt.figure()
        plt.imshow(img)
        plt.savefig(output_filename)
        plt.close()

        output_filename = os.path.join(split_output_folder, "samples_gen_sent_right.png")
        samples_right_gen_buffer = self.network.generate_images(100, from_node_path=node_path[1:], from_side='1')
        img = np.transpose(
            make_grid(samples_right_gen_buffer, nrow=int(np.sqrt(samples_right_gen_buffer.shape[0])),
                      padding=0).cpu().numpy(), (1, 2, 0))
        plt.figure()
        plt.imshow(img)
        plt.savefig(output_filename)
        plt.close()

        return



    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0],
                                            root_network_config.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            # with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        # Prepare settings for the training of HOLMES
        ## split trigger settings
        split_trigger = gr.config.update_config(training_config.split_trigger, gr.Config({"active": False}))
        if split_trigger["active"]:
            default_split_trigger = gr.Config()
            ## conditions
            default_split_trigger.conditions = gr.Config()
            default_split_trigger.conditions.min_init_n_epochs = 1
            default_split_trigger.conditions.n_epochs_min_between_splits = 1
            default_split_trigger.conditions.n_min_points = 0
            default_split_trigger.conditions.n_max_splits = 1e8
            ## fitness
            default_split_trigger.fitness_key = "total"
            ## type
            default_split_trigger.type = "plateau"
            default_split_trigger.parameters = gr.Config()
            ## train before split
            default_split_trigger.n_epochs_before_split = 0
            ## boundary config
            default_split_trigger.boundary_config = gr.Config()
            default_split_trigger.boundary_config.z_fitness = None
            default_split_trigger.boundary_config.algo = "cluster.KMeans"
            ## save
            default_split_trigger.save_model_before_after = False

            split_trigger = gr.config.update_config(split_trigger, default_split_trigger)

            ## parameters
            if split_trigger.type == "plateau":
                default_split_trigger.parameters.epsilon = 1
                default_split_trigger.parameters.n_steps_average = 5
            elif split_trigger.type == "threshold":
                default_split_trigger.parameters.threshold = 200
                default_split_trigger.parameters.n_max_bad_points = 100
            split_trigger.parameters = gr.config.update_config(split_trigger.parameters,
                                                               default_split_trigger.parameters)

        ## alternated training
        alternated_backward = gr.config.update_config(training_config.alternated_backward, gr.Config({"active": False}))
        if alternated_backward["active"]:
            alternated_backward.ratio_epochs = gr.config.update_config(alternated_backward.ratio_epochs, gr.Config({"connections": 1, "core": 9}))


        for epoch in range(training_config.n_epochs):
            # 1) check if elligible for split
            if split_trigger["active"]:
                self.trigger_split(train_loader, split_trigger)

            # 2) perform train epoch
            if (len(self.split_history) > 0) and (alternated_backward["active"]):
                if (self.n_epochs % int(alternated_backward["ratio_epochs"]["connections"]+alternated_backward["ratio_epochs"]["core"])) < alternated_backward["ratio_epochs"]["connections"]:
                    # train only connections
                    for leaf_path in self.network.get_leaf_pathes():
                        leaf_node = self.network.get_child_node(leaf_path)
                        for n, p in leaf_node.network.named_parameters():
                            if "_c" not in n:
                                p.requires_grad = False
                            else:
                                p.requires_grad = True
                else:
                    # train only children module without connections
                    for leaf_path in self.network.get_leaf_pathes():
                        leaf_node = self.network.get_child_node(leaf_path)
                        for n, p in leaf_node.network.named_parameters():
                            if "_c" not in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

            t0 = time.time()
            train_losses = self.train_epoch(train_loader, logger=logger)
            t1 = time.time()

            # update epoch counters
            self.n_epochs += 1
            for leaf_path in self.network.get_leaf_pathes():
                leaf_node = self.network.get_child_node(leaf_path)
                if hasattr(leaf_node, "epochs_since_split"):
                    leaf_node.epochs_since_split += 1
                else:
                    leaf_node.epochs_since_split = 1

            # log
            if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            # save model
            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

            if do_validation:
                # 3) Perform evaluation
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader, logger=logger)
                t3 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                if valid_losses:
                    valid_loss = valid_losses['total']
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

                # 4) Perform evaluation/test epoch
                if self.n_epochs % self.config.evaluation.save_results_every == 0:
                    save_results_per_node = (self.n_epochs == training_config.n_epochs)
                    self.evaluation_epoch(valid_loader, save_results_per_node=save_results_per_node)

            # 5) Stop splitting when close to the end
            # if split_trigger["active"] and (self.n_epochs >= (training_config.n_epochs - split_trigger.conditions.n_epochs_min_between_splits)):
            #     split_trigger["active"] = False



    def run_sequential_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        # Save the graph in the logger
        if logger is not None:
            root_network_config = self.config.node.network.parameters
            dummy_input = torch.FloatTensor(4, root_network_config.n_channels, root_network_config.input_size[0],
                                            root_network_config.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            # with torch.no_grad():
            #    logger.add_graph(self, dummy_input, verbose = False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        # Prepare settings for the training of HOLMES
        ## episodes setting
        n_episodes = training_config.n_episodes
        episodes_data_filter = training_config.episodes_data_filter
        n_epochs_per_episodes = training_config.n_epochs_per_episode
        blend_between_episodes = training_config.blend_between_episodes
        n_epochs_total = np.asarray(n_epochs_per_episodes).sum()

        ## split trigger settings
        split_trigger = gr.config.update_config(training_config.split_trigger, gr.Config({"active": False}))
        if split_trigger["active"]:
            default_split_trigger = gr.Config()
            ## conditions
            default_split_trigger.conditions = gr.Config()
            default_split_trigger.conditions.min_init_n_epochs = 1
            default_split_trigger.conditions.n_epochs_min_between_splits = 1
            default_split_trigger.conditions.n_min_points = 0
            default_split_trigger.conditions.n_max_splits = 1e8
            ## fitness
            default_split_trigger.fitness_key = "total"
            ## type
            default_split_trigger.type = "plateau"
            default_split_trigger.parameters = gr.Config()
            ## train before split
            default_split_trigger.n_epochs_before_split = 0
            ## boundary config
            default_split_trigger.boundary_config = gr.Config()
            default_split_trigger.boundary_config.z_fitness =  None
            default_split_trigger.boundary_config.algo =  "cluster.KMeans"
            ## save
            default_split_trigger.save_model_before_after = False

            split_trigger = gr.config.update_config(split_trigger, default_split_trigger)

            ## parameters
            if split_trigger.type == "plateau":
                default_split_trigger.parameters.epsilon = 1
                default_split_trigger.parameters.n_steps_average = 5
            elif split_trigger.type == "threshold":
                default_split_trigger.parameters.threshold = 200
                default_split_trigger.parameters.n_max_bad_points = 100
            split_trigger.parameters = gr.config.update_config(split_trigger.parameters, default_split_trigger.parameters)

        ## alternated training
        alternated_backward = gr.config.update_config(training_config.alternated_backward,
                                                      gr.Config({"active": False}))
        if alternated_backward["active"]:
            alternated_backward.ratio_epochs = gr.config.update_config(alternated_backward.ratio_epochs,
                                                                       gr.Config({"connections": 1, "core": 9}))
        ## experience replay
        experience_replay = gr.config.update_config(training_config.experience_replay, gr.Config({"active": False}))


        # prepare datasets per episodes
        train_images = [None] * n_episodes
        train_labels = [None] * n_episodes
        n_train_images = [None] * n_episodes
        train_weights = [None] * n_episodes
        train_datasets_ids =  [None] * n_episodes

        for episode_idx in range(n_episodes):
            cur_episode_data_filter = episodes_data_filter[episode_idx]
            cum_ratio = 0

            cur_episode_n_train_images = 0
            cur_episode_train_images = None
            cur_episode_train_labels = None
            cur_episode_train_datasets_ids = []

            for data_filter_idx, data_filter in enumerate(cur_episode_data_filter):

                cur_dataset_name = data_filter["dataset"]
                cur_filter = data_filter["filter"]
                cur_ratio = data_filter["ratio"]
                if data_filter_idx == len(cur_episode_data_filter) - 1:
                    if cur_ratio != 1.0 - cum_ratio:
                        raise ValueError("the sum of ratios per dataset in the episode must sum to one")
                cum_ratio += cur_ratio

                cur_train_dataset = train_loader.dataset.datasets[cur_dataset_name]
                cur_train_filtered_inds = do_filter_boolean(cur_train_dataset, cur_filter)
                cur_n_train_images = cur_train_filtered_inds.sum()
                if cur_episode_train_images is None:
                    cur_episode_train_images = cur_train_dataset.images[cur_train_filtered_inds]
                    cur_episode_train_labels = cur_train_dataset.labels[cur_train_filtered_inds]
                    cur_episode_train_weights = torch.tensor([cur_ratio / cur_n_train_images] * cur_n_train_images, dtype=torch.double)
                else:
                    cur_episode_train_images = torch.cat([cur_episode_train_images, cur_train_dataset.images[cur_train_filtered_inds]], dim=0)
                    cur_episode_train_labels = torch.cat([cur_episode_train_labels, cur_train_dataset.labels[cur_train_filtered_inds]], dim=0)
                    cur_episode_train_weights = torch.cat([cur_episode_train_weights, torch.tensor([cur_ratio / cur_n_train_images] * cur_n_train_images, dtype=torch.double)])
                cur_episode_n_train_images += cur_n_train_images
                cur_episode_train_datasets_ids += [cur_dataset_name] * cur_n_train_images

            train_images[episode_idx] = cur_episode_train_images
            train_labels[episode_idx] = cur_episode_train_labels
            n_train_images[episode_idx] = cur_episode_n_train_images
            train_weights[episode_idx] = cur_episode_train_weights
            train_datasets_ids[episode_idx] = cur_episode_train_datasets_ids

        # /!\ test images will be accumulated in the dataloader at each episode to test forgetting on previously seen images
        if do_validation:
            valid_images = [None] * n_episodes
            valid_labels = [None] * n_episodes
            n_valid_images = [None] * n_episodes
            valid_datasets_ids = [None] * n_episodes

            for episode_idx in range(n_episodes):
                cur_episode_data_filter = episodes_data_filter[episode_idx]
                cum_ratio = 0

                cur_episode_n_valid_images = 0
                cur_episode_valid_images = None
                cur_episode_valid_labels = None
                cur_episode_valid_datasets_ids = []

                for data_filter_idx, data_filter in enumerate(cur_episode_data_filter):
                    cur_dataset_name = data_filter["dataset"]
                    cur_filter = data_filter["filter"]
                    cur_ratio = data_filter["ratio"]
                    if data_filter_idx == len(cur_episode_data_filter) - 1:
                        if cur_ratio != 1.0 - cum_ratio:
                            raise ValueError("the sum of ratios per dataset in the episode must sum to one")
                    cum_ratio += cur_ratio

                    cur_valid_dataset = valid_loader.dataset.datasets[cur_dataset_name]
                    cur_valid_filtered_inds = do_filter_boolean(cur_valid_dataset, cur_filter)
                    cur_n_valid_images = cur_valid_filtered_inds.sum()
                    if cur_episode_valid_images is None:
                        cur_episode_valid_images = cur_valid_dataset.images[cur_valid_filtered_inds]
                        cur_episode_valid_labels = cur_valid_dataset.labels[cur_valid_filtered_inds]
                    else:
                        cur_episode_valid_images = torch.cat([cur_episode_valid_images, cur_valid_dataset.images[cur_valid_filtered_inds]], dim=0)
                        cur_episode_valid_labels = torch.cat([cur_episode_valid_labels, cur_valid_dataset.labels[cur_valid_filtered_inds]], dim=0)
                    cur_episode_n_valid_images += cur_n_valid_images
                    cur_episode_valid_datasets_ids += [cur_dataset_name] * cur_n_valid_images

                valid_images[episode_idx] = cur_episode_valid_images
                valid_labels[episode_idx] = cur_episode_valid_labels
                n_valid_images[episode_idx] = cur_episode_n_valid_images
                valid_datasets_ids[episode_idx] = cur_episode_valid_datasets_ids

        for episode_idx in range(n_episodes):
            # accumulate over test set
            if do_validation:
                cum_n_valid_images = valid_loader.dataset.n_images + n_valid_images[episode_idx]
                cum_valid_images = torch.cat([valid_loader.dataset.images, valid_images[episode_idx]], dim=0)
                cum_valid_labels = torch.cat([valid_loader.dataset.labels, valid_labels[episode_idx]], dim=0)
                cum_valid_datasets_ids = valid_loader.dataset.datasets_ids + valid_datasets_ids[episode_idx]
                valid_loader.dataset.update(cum_n_valid_images, cum_valid_images, cum_valid_labels, cum_valid_datasets_ids)

            for epoch in range(n_epochs_per_episodes[episode_idx]):
                train_loader.dataset.update(n_train_images[episode_idx], train_images[episode_idx],
                                            train_labels[episode_idx], train_datasets_ids[episode_idx])
                train_loader.sampler.num_samples = len(train_weights[episode_idx])
                train_loader.sampler.weights = train_weights[episode_idx]

                #1) Prepare DataLoader by blending with prev/next episode
                if blend_between_episodes["active"]:
                    cur_epoch_images = deepcopy(train_images[episode_idx])
                    cur_epoch_labels = deepcopy(train_labels[episode_idx])
                    cur_epoch_datasets_ids = deepcopy(train_datasets_ids[episode_idx])
                    # blend images from previous experiment
                    blend_with_prev_episode = blend_between_episodes["blend_with_prev"]
                    if episode_idx > 0 and blend_with_prev_episode["active"]:
                        cur_episode_fraction = float(epoch / n_epochs_per_episodes[episode_idx])
                        if cur_episode_fraction < blend_with_prev_episode["time_fraction"]:
                            if blend_with_prev_episode["blend_type"] == "linear":
                                blend_prev_proportion = - 0.5 / blend_with_prev_episode[
                                    "time_fraction"] * cur_episode_fraction + 0.5
                                n_data_from_prev = int(blend_prev_proportion * n_train_images[episode_idx])
                                ids_to_take_from_prev_episode = torch.multinomial(train_weights[episode_idx - 1],
                                                                                  n_data_from_prev, True)
                                ids_to_replace_in_cur_episode = torch.multinomial(train_weights[episode_idx],
                                                                                  n_data_from_prev, False)
                                for data_idx in range(n_data_from_prev):
                                    cur_epoch_images[ids_to_replace_in_cur_episode[data_idx]] = train_images[episode_idx - 1][
                                        ids_to_take_from_prev_episode[data_idx]]
                                    cur_epoch_labels[ids_to_replace_in_cur_episode[data_idx]] = train_labels[episode_idx - 1][
                                        ids_to_take_from_prev_episode[data_idx]]
                                    cur_epoch_datasets_ids[ids_to_replace_in_cur_episode[data_idx]] = train_datasets_ids[episode_idx - 1][
                                        ids_to_take_from_prev_episode[data_idx]]
                            else:
                                raise NotImplementedError("only linear blending is implemented")
                    # blend images from next experiment
                    blend_with_next_episode = blend_between_episodes["blend_with_next"]
                    if episode_idx < n_episodes - 1 and blend_with_next_episode["active"]:
                        cur_episode_fraction = float(epoch / n_epochs_per_episodes[episode_idx])
                        if cur_episode_fraction > (1.0 - blend_with_next_episode["time_fraction"]):
                            if blend_with_next_episode["blend_type"] == "linear":
                                blend_next_proportion = 0.5 / blend_with_prev_episode[
                                    "time_fraction"] * cur_episode_fraction \
                                                        + 0.5 - (0.5 / blend_with_prev_episode["time_fraction"])
                                n_data_from_next = int(blend_next_proportion * n_train_images[episode_idx])
                                ids_to_take_from_next_episode = torch.multinomial(train_weights[episode_idx + 1],
                                                                                  n_data_from_next, True)
                                ids_to_replace_in_cur_episode = torch.multinomial(train_weights[episode_idx],
                                                                                  n_data_from_next, False)
                                for data_idx in range(n_data_from_next):
                                    cur_epoch_images[ids_to_replace_in_cur_episode[data_idx]] = train_images[episode_idx + 1][
                                        ids_to_take_from_next_episode[data_idx]]
                                    cur_epoch_labels[ids_to_replace_in_cur_episode[data_idx]] = train_labels[episode_idx + 1][
                                        ids_to_take_from_next_episode[data_idx]]
                                    cur_epoch_datasets_ids[ids_to_replace_in_cur_episode[data_idx]] = \
                                    train_datasets_ids[episode_idx - 1][
                                        ids_to_take_from_next_episode[data_idx]]
                            else:
                                raise NotImplementedError("only linear blending is implemented")

                    train_loader.dataset.update(n_train_images[episode_idx], cur_epoch_images, cur_epoch_labels, cur_epoch_datasets_ids)

                # 2) Prepare DataLoader by adding replay images
                if experience_replay['active']:
                    if (self.n_epochs > experience_replay.min_init_n_epochs) and (self.n_epochs % experience_replay.generate_every == 0):
                        # replay
                        n_replay = int(experience_replay.percent / (1-experience_replay.percent) * n_train_images[episode_idx])
                        n_replay_per_leaf_node = int(n_replay / len(self.network.get_leaf_pathes()))
                        cur_epoch_n_images = train_loader.dataset.n_images
                        cur_epoch_images = train_loader.dataset.images
                        cur_epoch_labels = train_loader.dataset.labels
                        cur_epoch_datasets_ids = train_loader.dataset.datasets_ids
                        for leaf_path in self.network.get_leaf_pathes():
                            cur_gen_images = self.network.generate_images(n_replay_per_leaf_node, from_node_path=leaf_path[1:-1], from_side=leaf_path[-1]).cpu()
                            cur_epoch_n_images += n_replay_per_leaf_node
                            cur_epoch_images = torch.cat([cur_epoch_images, cur_gen_images])
                            cur_epoch_labels = torch.cat([cur_epoch_labels, torch.LongTensor([-1] * len(cur_gen_images))])
                            cur_epoch_datasets_ids = cur_epoch_datasets_ids + [None] *  len(cur_gen_images)

                        n_gen_images = cur_epoch_n_images - train_loader.dataset.n_images
                        cur_epoch_train_weights = torch.cat([train_loader.sampler.weights*train_loader.dataset.n_images/cur_epoch_n_images,  torch.tensor([experience_replay.percent / n_gen_images] * n_gen_images, dtype=torch.double)])
                        train_loader.sampler.num_samples = cur_epoch_n_images
                        train_loader.sampler.weights = cur_epoch_train_weights
                        train_loader.dataset.update(cur_epoch_n_images, cur_epoch_images, cur_epoch_labels, cur_epoch_datasets_ids)


                # 2) check if elligible for split
                if split_trigger["active"]:
                    self.trigger_split(train_loader, split_trigger)

                # 3) Perform train epoch
                t0 = time.time()
                train_losses = self.train_epoch(train_loader, logger=logger)
                t1 = time.time()

                # update epoch counters
                self.n_epochs += 1
                for leaf_path in self.network.get_leaf_pathes():
                    leaf_node = self.network.get_child_node(leaf_path)
                    if hasattr(leaf_node, "epochs_since_split"):
                        leaf_node.epochs_since_split += 1
                    else:
                        leaf_node.epochs_since_split = 1

                # logging
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in train_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                    logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                    self.n_epochs)

                # save model
                if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'curent_weight_model.pth'))
                if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

                if do_validation:
                    # 4) Perform validation epoch
                    t2 = time.time()
                    valid_losses = self.valid_epoch(valid_loader, logger=logger)
                    t3 = time.time()
                    if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                        for k, v in valid_losses.items():
                            logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                        logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                        self.n_epochs)

                    if valid_losses:
                        valid_loss = valid_losses['total']
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

                    # 5) Perform evaluation/test epoch
                    if self.n_epochs % self.config.evaluation.save_results_every == 0:
                        save_results_per_node = (self.n_epochs == n_epochs_total)
                        self.evaluation_epoch(valid_loader, save_results_per_node=save_results_per_node)

                # 6) Stop splitting when close to the end
                if self.n_epochs >= (n_epochs_total - split_trigger.conditions.n_epochs_min_between_splits):
                    split_trigger["active"] = False

    def trigger_split(self, train_loader, split_trigger):

        splitted_leafs = []

        if (len(self.split_history) > split_trigger.conditions.n_max_splits) or (
                self.n_epochs < split_trigger.conditions.min_init_n_epochs):
            return

        self.eval()
        train_fitness = None
        taken_pathes = []
        x_ids = []
        labels = []

        old_augment_state = train_loader.dataset.data_augmentation
        train_loader.dataset.data_augmentation = False

        with torch.no_grad():
            for data in train_loader:
                x = data["obs"]
                x = self.push_variable_to_device(x)
                x_ids.append(data["index"])
                labels.append(data["label"])
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                cur_train_fitness = batch_losses[split_trigger.fitness_key]
                # save losses
                if train_fitness is None:
                    train_fitness = np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)
                else:
                    train_fitness = np.vstack(
                        [train_fitness, np.expand_dims(cur_train_fitness.detach().cpu().numpy(), axis=-1)])
                # save taken pathes
                taken_pathes += model_outputs["path_taken"]

            x_ids = torch.cat(x_ids)
            labels = torch.cat(labels)

        for leaf_path in list(set(taken_pathes)):
            leaf_node = self.network.get_child_node(leaf_path)
            leaf_x_ids = (np.array(taken_pathes, copy=False) == leaf_path)
            generated_ids_in_leaf_x_ids = (np.asarray(labels[leaf_x_ids]).squeeze() == -1)
            leaf_n_real_points = leaf_x_ids.sum() - generated_ids_in_leaf_x_ids.sum()
            split_x_fitness = train_fitness[leaf_x_ids, :]
            split_x_fitness[generated_ids_in_leaf_x_ids] = 0
            leaf_node.fitness_last_epochs.append(split_x_fitness[~generated_ids_in_leaf_x_ids].mean())

            if (leaf_node.epochs_since_split < split_trigger.conditions.n_epochs_min_between_splits) or (leaf_n_real_points < split_trigger.conditions.n_min_points):
                continue;

            trigger_split_in_leaf = False
            if split_trigger.type == "threshold":
                poor_buffer = (split_x_fitness > split_trigger.parameters.threshold).squeeze()
                if (poor_buffer.sum() > split_trigger.parameters.n_max_bad_points):
                    trigger_split_in_leaf = True

            elif split_trigger.type == "plateau":
                if len(leaf_node.fitness_last_epochs) > split_trigger.parameters.n_steps_average:
                    leaf_node.fitness_last_epochs.pop(0)
                fitness_vals = np.asarray(leaf_node.fitness_last_epochs)
                fitness_speed_last_epochs = fitness_vals[1:] - fitness_vals[:-1]
                running_average_speed = np.abs(fitness_speed_last_epochs.mean())
                if (running_average_speed < split_trigger.parameters.epsilon):
                    trigger_split_in_leaf = True

            if trigger_split_in_leaf:
                # Split Node
                split_dataset_config = train_loader.dataset.config
                split_dataset_config.datasets = {}
                split_dataset_config.data_augmentation = False
                split_dataset = MIXEDDataset(config=split_dataset_config)

                split_x_ids = x_ids[leaf_x_ids]
                n_seen_images = len(split_x_ids)
                split_seen_images = train_loader.dataset.images[split_x_ids]
                split_seen_labels = train_loader.dataset.labels[split_x_ids]

                split_dataset.update(n_seen_images, split_seen_images, split_seen_labels)
                split_loader = DataLoader(split_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=0)

                self.split_node(leaf_path,  split_trigger, x_loader=split_loader, x_fitness=split_x_fitness)

                del split_seen_images, split_seen_labels, split_dataset, split_loader
                # update counters
                leaf_node.epochs_since_split = None
                leaf_node.fitness_speed_last_epoches = []

                splitted_leafs.append(leaf_path)

                # uncomment following line for allowing only one split at the time
                # break;

        train_loader.dataset.data_augmentation = old_augment_state
        return splitted_leafs


    def train_epoch(self, train_loader, logger=None):
        self.train()

        taken_pathes = []
        losses = {}

        for data in train_loader:
            x = data['obs']
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
            # g = make_dot(model_outputs, params=dict(self.network.named_parameters()))
            # g.view()
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs, reduction="none")
            # backward
            loss = batch_losses['total'].mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                else:
                    losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])
            # save taken path
            taken_pathes += model_outputs["path_taken"]


        # Logger save results per leaf
        if logger is not None:
            for leaf_path in list(set(taken_pathes)):
                if len(leaf_path) > 1:
                    leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]
                    for k, v in losses.items():
                        leaf_v = v[leaf_x_ids, :]
                        logger.add_scalars('loss/{}'.format(k), {'train-{}'.format(leaf_path): np.mean(leaf_v)},
                                           self.n_epochs)

        # Average loss on all tree
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        images = None
        recon_images = None
        embeddings = None
        labels = None
        if logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0:
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embeddings = []
                labels = []
                if images is None:
                    images = []


        taken_pathes = []

        with torch.no_grad():
            for data in valid_loader:
                x = data['obs']
                x = self.push_variable_to_device(x)
                y = data['label'].squeeze()
                y = self.push_variable_to_device(y)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs, reduction="none")
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])
                # record embeddings
                if record_valid_images:
                    for i in range(len(x)):
                        images.append(x[i].unsqueeze(0))
                    for i in range(len(x)):
                        recon_images.append(model_outputs["recon_x"][i].unsqueeze(0))
                if record_embeddings:
                    for i in range(len(x)):
                        embeddings.append(model_outputs["z"][i].unsqueeze(0))
                        labels.append(data['label'][i].unsqueeze(0))
                    if not record_valid_images:
                        for i in range(len(x)):
                            images.append(x[i].unsqueeze(0))

                taken_pathes += model_outputs["path_taken"]


        # 2) LOGGER SAVE RESULT PER LEAF
        for leaf_path in list(set(taken_pathes)):
            leaf_x_ids = np.where(np.array(taken_pathes, copy=False) == leaf_path)[0]

            if record_embeddings:
                leaf_embeddings = torch.cat([embeddings[i] for i in leaf_x_ids])
                leaf_labels = torch.cat([labels[i] for i in leaf_x_ids])
                leaf_images = torch.cat([images[i] for i in leaf_x_ids])
                leaf_images = tensorboardhelper.resize_embeddings(leaf_images)
                try:
                    logger.add_embedding(
                        leaf_embeddings,
                        metadata=leaf_labels,
                        label_img=leaf_images,
                        global_step=self.n_epochs,
                        tag="leaf_{}".format(leaf_path))
                except:
                    pass

            if record_valid_images:
                n_images = min(len(leaf_x_ids), 40)
                sampled_ids = np.random.choice(len(leaf_x_ids), n_images, replace=False)
                input_images = torch.cat([images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                output_images = torch.cat([recon_images[i] for i in leaf_x_ids[sampled_ids]]).cpu()
                if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                    output_images = torch.sigmoid(output_images)
                vizu_tensor_list = [None] * (2 * n_images)
                vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
                vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
                img = make_grid(vizu_tensor_list, nrow=2, padding=0)
                try:
                    logger.add_image("leaf_{}".format(leaf_path), img, self.n_epochs)
                except:
                    pass

        # 4) AVERAGE LOSS ON WHOLE TREE AND RETURN
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def evaluation_epoch(self, test_loader, save_results_per_node=False):
        self.eval()
        losses = {}

        n_images = len(test_loader.dataset)
        if "RandomSampler" in test_loader.sampler.__class__.__name__:
            warnings.warn("WARNING: evaluation is performed on shuffled test dataloader")
        tree_depth = 1
        for split_k, split in self.split_history.items():
            if (split["depth"]+2) > tree_depth:
                tree_depth = (split["depth"]+2)

        test_results = {"taken_pathes": [None] * n_images, "labels": np.empty(n_images, dtype=np.int),
                        "nll": np.empty(n_images, dtype=np.float), "recon_loss": np.empty(n_images, dtype=np.float),
                        "cluster_classification_pred": np.empty(n_images, dtype=np.int),
                        "cluster_classification_acc": np.empty(n_images, dtype=np.bool),
                        "5NN_classification_pred": np.empty(n_images, dtype=np.int),
                        "5NN_classification_err": np.empty(n_images, dtype=np.bool),
                        "10NN_classification_pred": np.empty(n_images, dtype=np.int),
                        "10NN_classification_err": np.empty(n_images, dtype=np.bool),
                        "20NN_classification_pred": np.empty(n_images, dtype=np.int),
                        "20NN_classification_err": np.empty(n_images, dtype=np.bool)}
        #TODO: "spread" kNN

        test_results_per_node = {}
        for node_path in self.network.get_node_pathes():
            test_results_per_node[node_path] = dict()
            test_results_per_node[node_path]["x_ids"] = []
            test_results_per_node[node_path]["z"] = []
            test_results_per_node[node_path]["recon_x"] = []
            test_results_per_node[node_path]["nll"] = []
            test_results_per_node[node_path]["recon_loss"] = []
            test_results_per_node[node_path]["cluster_classification_pred"] = []
            test_results_per_node[node_path]["cluster_classification_acc"] = []
            test_results_per_node[node_path]["5NN_classification_pred"] = []
            test_results_per_node[node_path]["5NN_classification_err"] = []
            test_results_per_node[node_path]["10NN_classification_pred"] = []
            test_results_per_node[node_path]["10NN_classification_err"] = []
            test_results_per_node[node_path]["20NN_classification_pred"] = []
            test_results_per_node[node_path]["20NN_classification_err"] = []

        with torch.no_grad():
            # Compute results per image
            idx_offset = 0
            for data in test_loader:
                x = data['obs']
                x = self.push_variable_to_device(x)
                cur_x_ids = np.asarray(data["index"], dtype=np.int)
                y = data['label'].squeeze()
                test_results["labels"][cur_x_ids] = y.detach().cpu().numpy()

                # forward
                all_nodes_outputs = self.network.depth_first_forward_whole_branch_preorder(x)
                for node_idx in range(len(all_nodes_outputs)):
                    cur_node_path = all_nodes_outputs[node_idx][0][0]
                    cur_node_x_ids = np.asarray(all_nodes_outputs[node_idx][1], dtype=np.int)
                    cur_node_outputs = all_nodes_outputs[node_idx][2]
                    loss_inputs = {key: cur_node_outputs[key] for key in self.loss_f.input_keys_list}
                    cur_losses = self.loss_f(loss_inputs, reduction="none")
                    test_results_per_node[cur_node_path]["x_ids"] += list(cur_x_ids[cur_node_x_ids])
                    test_results_per_node[cur_node_path]["z"] += list(cur_node_outputs["z"].detach().cpu().numpy())
                    test_results_per_node[cur_node_path]["recon_x"] += list(cur_node_outputs["recon_x"].detach().cpu().numpy())
                    test_results_per_node[cur_node_path]["nll"] += list(cur_losses["total"].detach().cpu().numpy())
                    test_results_per_node[cur_node_path]["recon_loss"] += list(cur_losses["recon"].detach().cpu().numpy())

                    cur_node = self.network.get_child_node(cur_node_path)
                    if cur_node.leaf:
                        for x_idx in cur_node_x_ids:
                            test_results["taken_pathes"][cur_x_ids[x_idx]] = cur_node_path
                        test_results["nll"][cur_x_ids[cur_node_x_ids]] = cur_losses["total"].detach().cpu().numpy()
                        test_results["recon_loss"][cur_x_ids[cur_node_x_ids]] = cur_losses["recon"].detach().cpu().numpy()

        # compute results for classification
        for node_path in self.network.get_node_pathes():
            # update lists to numpy
            test_results_per_node[node_path]["x_ids"] = np.asarray(test_results_per_node[node_path]["x_ids"], dtype = np.int)
            test_results_per_node[node_path]["z"] = np.asarray(test_results_per_node[node_path]["z"], dtype = np.float)
            test_results_per_node[node_path]["recon_x"] = np.asarray(test_results_per_node[node_path]["recon_x"], dtype=np.float)
            test_results_per_node[node_path]["nll"] = np.asarray(test_results_per_node[node_path]["nll"],
                                                                     dtype=np.float)
            test_results_per_node[node_path]["recon_loss"] = np.asarray(test_results_per_node[node_path]["recon_loss"],
                                                                     dtype=np.float)

            # cluster classification accuracy
            labels_in_node = test_results["labels"][test_results_per_node[node_path]["x_ids"]]
            majority_voted_class = -1
            max_n_votes = 0
            for label in list(set(labels_in_node)):
                label_count = (labels_in_node == label).sum()
                if label_count > max_n_votes:
                    max_n_votes = label_count
                    majority_voted_class = label
            cur_node_x_ids = test_results_per_node[node_path]["x_ids"]
            test_results_per_node[node_path]["cluster_classification_pred"] = np.asarray([majority_voted_class] * len(cur_node_x_ids), dtype=np.int)
            test_results_per_node[node_path]["cluster_classification_acc"] = (test_results["labels"][cur_node_x_ids] == majority_voted_class)

            # k-NN classification accuracy
            for x_idx in range(len(cur_node_x_ids)):
                distances_to_point_in_node = np.linalg.norm(test_results_per_node[node_path]["z"][x_idx] - test_results_per_node[node_path]["z"], axis=1)
                closest_point_ids = np.argpartition(distances_to_point_in_node, min(20, len(distances_to_point_in_node)-1))
                # remove curr_idx from closest point
                closest_point_ids = np.delete(closest_point_ids, np.where(closest_point_ids == x_idx)[0])
                for k_idx, k in enumerate([5,10,20]):
                    voting_labels = test_results["labels"][closest_point_ids[:k]]
                    majority_voted_class = -1
                    max_n_votes = 0
                    for label in list(set(voting_labels)):
                        label_count = (voting_labels == label).sum()
                        if label_count > max_n_votes:
                            max_n_votes = label_count
                            majority_voted_class = label
                    test_results_per_node[node_path]["{}NN_classification_pred".format(k)].append(majority_voted_class)
                    test_results_per_node[node_path]["{}NN_classification_err".format(k)].append(test_results["labels"][x_idx] != majority_voted_class)

            for k_idx, k in enumerate([5, 10, 20]):
                test_results_per_node[node_path]["{}NN_classification_pred".format(k)] = np.asarray(test_results_per_node[node_path]["{}NN_classification_pred".format(k)], dtype=np.int)
                test_results_per_node[node_path]["{}NN_classification_err".format(k)] = np.asarray(test_results_per_node[node_path]["{}NN_classification_err".format(k)], dtype=np.bool)

            node = self.network.get_child_node(node_path)
            if node.leaf:
                test_results["cluster_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["cluster_classification_pred"]
                test_results["cluster_classification_acc"][cur_node_x_ids] = test_results_per_node[node_path]["cluster_classification_acc"]
                test_results["5NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["5NN_classification_pred"]
                test_results["5NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["5NN_classification_err"]
                test_results["10NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["10NN_classification_pred"]
                test_results["10NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["10NN_classification_err"]
                test_results["20NN_classification_pred"][cur_node_x_ids] = test_results_per_node[node_path]["20NN_classification_pred"]
                test_results["20NN_classification_err"][cur_node_x_ids] = test_results_per_node[node_path]["20NN_classification_err"]



        # Save results
        if not os.path.exists(os.path.join(self.config.evaluation.folder, "test_results")):
            os.makedirs(os.path.join(self.config.evaluation.folder, "test_results"))
        np.savez(os.path.join(self.config.evaluation.folder, "test_results",
                              "test_results_epoch_{}.npz".format((self.n_epochs))), **test_results)
        if save_results_per_node:
            np.savez(os.path.join(self.config.evaluation.folder, "test_results",
                                  "test_results_per_node_epoch_{}.npz".format((self.n_epochs))), **test_results_per_node)
        return

    def get_encoder(self):
        pass

    def get_decoder(self):
        pass

    def save_checkpoint(self, checkpoint_filepath):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "split_history": self.split_history
        }

        torch.save(network, checkpoint_filepath)


""" =========================================================================================
CONNECTED MODULES 
==========================================================================================="""


class ConnectedEncoder(gr.dnn.networks.encoders.BaseDNNEncoder):
    def __init__(self, encoder_instance, depth, connect_lf=False, connect_gf=False, **kwargs):
        gr.dnn.networks.encoders.BaseDNNEncoder.__init__(self, config=encoder_instance.config)
        # connections and depth in the tree (number of connections)
        self.connect_lf = connect_lf
        self.connect_gf = connect_gf
        self.depth = depth

        # copy parent network layers
        self.lf = encoder_instance.lf
        self.gf = encoder_instance.gf
        self.ef = encoder_instance.ef

        # add lateral connections
        ## lf
        if self.connect_lf:
            if self.lf.out_connection_type[0] == "conv":
                connection_channels = self.lf.out_connection_type[1]
                self.lf_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.lf.out_connection_type[0] == "lin":
                connection_dim = self.lf.out_connection_type[1]
                self.lf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## gf
        if self.connect_gf:
            if self.gf.out_connection_type[0] == "conv":
                connection_channels = self.gf.out_connection_type[1]
                self.gf_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gf.out_connection_type[0] == "lin":
                connection_dim = self.gf.out_connection_type[1]
                self.gf_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        initialization_net = initialization.get_initialization("kaiming_uniform")
        self.gf.apply(initialization_net)
        self.ef.apply(initialization_net)
        initialization_c = initialization.get_initialization("uniform")
        if self.connect_lf:
            self.lf_c.apply(initialization_c)
        if self.connect_gf:
            self.gf_c.apply(initialization_c)

    """
        #initialization_net = initialization.get_initialization("null")
        #self.gf.apply(initialization_net)
        #self.ef.apply(initialization_net)
        #initialization_c = initialization.get_initialization("connections_identity")
        initialization_c = initialization.get_initialization("null")
        if self.connect_lf:
            self.lf_c.apply(initialization_c)
        if self.connect_gf:
            self.gf_c.apply(initialization_c)
        """

    def forward(self, x, parent_lf=None, parent_gf=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf += self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf += self.gf_c(parent_gf)

        # encoding
        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                mu = mu.squeeze(dim=-1).squeeze(dim=-1)
                logvar = logvar.squeeze(dim=-1).squeeze(dim=-1)
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z, "mu": mu, "logvar": logvar}
        elif self.config.encoder_conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs = {"x": x, "lf": lf, "gf": gf, "z": z}

        return encoder_outputs

    def forward_for_graph_tracing(self, x, parent_lf=None, parent_gf=None):
        # local feature map
        lf = self.lf(x)
        # add the connections
        if self.connect_lf:
            lf += self.lf_c(parent_lf)

        # global feature map
        gf = self.gf(lf)
        # add the connections
        if self.connect_gf:
            gf += self.gf_c(parent_gf)

        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf


class ConnectedDecoder(gr.dnn.networks.decoders.BaseDNNDecoder):
    def __init__(self, decoder_instance, depth, connect_gfi=False, connect_lfi=False, connect_recon=False, **kwargs):
        gr.dnn.networks.decoders.BaseDNNDecoder.__init__(self, config=decoder_instance.config)

        # connections and depth in the tree (number of connections)
        self.connect_gfi = connect_gfi
        self.connect_lfi = connect_lfi
        self.connect_recon = connect_recon
        self.depth = depth

        # copy parent network layers
        self.efi = decoder_instance.efi
        self.gfi = decoder_instance.gfi
        self.lfi = decoder_instance.lfi

        # add lateral connections
        ## gfi
        if self.connect_gfi:
            if self.efi.out_connection_type[0] == "conv":
                connection_channels = self.efi.out_connection_type[1]
                self.gfi_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.efi.out_connection_type[0] == "lin":
                connection_dim = self.efi.out_connection_type[1]
                self.gfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())
        ## lfi
        if self.connect_lfi:
            if self.gfi.out_connection_type[0] == "conv":
                connection_channels = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False), nn.ReLU())
            elif self.gfi.out_connection_type[0] == "lin":
                connection_dim = self.gfi.out_connection_type[1]
                self.lfi_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())

        ## lfi
        if self.connect_recon:
            if self.lfi.out_connection_type[0] == "conv":
                connection_channels = self.lfi.out_connection_type[1]
                self.recon_c = nn.Sequential(nn.Conv2d(connection_channels, connection_channels, kernel_size=1, stride=1, bias = False))
            elif self.lfi.out_connection_type[0] == "lin":
                connection_dim = self.lfi.out_connection_type[1]
                self.recon_c = nn.Sequential(nn.Linear(connection_dim, connection_dim), nn.ReLU())


        initialization_net = initialization.get_initialization("kaiming_uniform")
        self.efi.apply(initialization_net)
        self.gfi.apply(initialization_net)
        self.lfi.apply(initialization_net)
        initialization_c = initialization.get_initialization("uniform")
        if self.connect_gfi:
            self.gfi_c.apply(initialization_c)
        if self.connect_lfi:
            self.lfi_c.apply(initialization_c)
        if self.connect_recon:
            self.recon_c.apply(initialization_c)

        """
        #initialization_net = initialization.get_initialization("null")
        #self.efi.apply(initialization_net)
        #self.gfi.apply(initialization_net)
        #self.lfi.apply(initialization_net)
        #initialization_c = initialization.get_initialization("connections_identity")
        initialization_c = initialization.get_initialization("null")
        if self.connect_gfi:
            self.gfi_c.apply(initialization_c)
        if self.connect_lfi:
            self.lfi_c.apply(initialization_c)
        if self.connect_recon:
            self.recon_c.apply(initialization_c)
        """

    def forward(self, z, parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)

        if z.dim() == 2 and type(self).__name__ == "DumoulinDecoder":  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi += self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi += self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x += self.recon_c(parent_recon_x)

        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs

    def forward_for_graph_tracing(self, z,  parent_gfi=None, parent_lfi=None, parent_recon_x=None):
        if z.dim() == 2:  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # global feature map
        gfi = self.efi(z)
        # add the connections
        if self.connect_gfi:
            gfi += self.gfi_c(parent_gfi)

        # local feature map
        lfi = self.gfi(gfi)
        # add the connections
        if self.connect_lfi:
            lfi += self.lfi_c(parent_lfi)

        # recon_x
        recon_x = self.lfi(lfi)
        # add the connections
        if self.connect_recon:
            recon_x += self.recon_c(parent_recon_x)

        return recon_x
