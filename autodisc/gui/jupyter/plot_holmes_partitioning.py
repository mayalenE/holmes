import autodisc as ad
import plotly
import zipfile
import os
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import warnings
import random
from autodisc.gui.jupyter.misc import create_colormap, transform_image_from_colormap, lighten_color


def get_diversity_representants(dataset_embeddings, n_candidates, n_images):
    n_points = dataset_embeddings.shape[0]
    dim_embedding = dataset_embeddings.shape[1]

    candidate_ids = np.random.randint(low=0, high=n_points, size=(n_candidates * n_images))
    candidate_embeddings = dataset_embeddings[candidate_ids, :].reshape([n_candidates, n_images, dim_embedding])
    candidate_ids = candidate_ids.reshape([n_candidates, n_images])

    dists = []
    for j in range(n_images):
        for k in range(n_images):
            dists.append(np.linalg.norm(candidate_embeddings[:, j, :] - candidate_embeddings[:, k, :], axis=1))
    dists = np.array(dists) / np.sqrt(dim_embedding)

    # Standardized mean dispersion
    Mprime = dists.sum(axis=0) / (n_images * (n_images - 1))

    # Standardized eveness of dispersion
    q = 2
    H = np.power(np.power(dists / dists.sum(axis=0), q).sum(axis=0), 1 / (1 - q))
    E = (1 + np.sqrt(1 + 4 * H)) / (2 * n_images)

    # functional trait diversity
    diversity = 1 + (n_images - 1) * E * Mprime

    indices_diverse = candidate_ids[np.argmax(diversity)]


    return indices_diverse



def plot_holmes_partitioning(experiment_definition, repetition_id=0, experiment_statistics=None, data_filters=None,
                             config=None, **kwargs):
    plotly_default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    plotly_default_colors_darker = [
        lighten_color([float(cc) / 255.0 for cc in c.replace("rgb(", "").replace(")", "").split(", ")], 1.4) for c in
        plotly.colors.DEFAULT_PLOTLY_COLORS]
    plotly_default_colors_darker = ["rgb({},{},{})".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in
                                    plotly_default_colors_darker]
    default_config = dict(
        random_seed=0,

        # global style config
        global_layout=dict(
            height=700,
            width=700,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            xaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            title=dict(
                text='',
                font=dict(
                    color="black",
                    size=22,
                    family='Times New Roman'
                )
            ),
            hovermode='closest',
            showlegend=True,
        ),

        # Shapes style config
        shapes=dict(
            line=dict(
                width=2
            ),
            layer="below"
        ),
        shapes_background_colors=plotly_default_colors,
        shapes_lines_colors=plotly_default_colors_darker,

        # Images style config
        margin_x=1,
        margin_y=1,
        images=dict(
            xref="x",
            yref="y",
            sizex=10,
            sizey=10,
            opacity=1,
            xanchor="center",
            yanchor="middle",
            layer="above"
        ),
        images_transform=True,
        images_transform_colormaps=[
            create_colormap(np.array(
                [[255, 255, 255], [119, 255, 255], [23, 223, 252], [0, 190, 250], [0, 158, 249], [0, 142, 249],
                 [81, 125, 248], [150, 109, 248], [192, 77, 247], [232, 47, 247], [255, 9, 247],
                 [200, 0, 84]]) / 255 * 8),
            # WBPR
            create_colormap(np.array(
                [[0, 0, 4], [0, 0, 8], [0, 4, 8], [0, 8, 8], [4, 8, 4], [8, 8, 0], [8, 4, 0], [8, 0, 0], [4, 0, 0]])),
            # BCYR
            create_colormap(np.array(
                [[0, 2, 0], [0, 4, 0], [4, 6, 0], [8, 8, 0], [8, 4, 4], [8, 0, 8], [4, 0, 8], [0, 0, 8], [0, 0, 4]])),
            # GYPB
            create_colormap(np.array(
                [[4, 0, 2], [8, 0, 4], [8, 0, 6], [8, 0, 8], [4, 4, 4], [0, 8, 0], [0, 6, 0], [0, 4, 0], [0, 2, 0]])),
            # PPGG
            create_colormap(np.array([[4, 4, 6], [2, 2, 4], [2, 4, 2], [4, 6, 4], [6, 6, 4], [4, 2, 2]])),  # BGYR
            create_colormap(np.array([[4, 6, 4], [2, 4, 2], [4, 4, 2], [6, 6, 4], [6, 4, 6], [2, 2, 4]])),  # GYPB
            create_colormap(np.array([[6, 6, 4], [4, 4, 2], [4, 2, 4], [6, 4, 6], [4, 6, 6], [2, 4, 2]])),  # YPCG
            create_colormap(np.array([[8, 8, 8], [7, 7, 7], [5, 5, 5], [3, 3, 3], [0, 0, 0]]), is_marker_w=False),
            # W/B
            create_colormap(np.array([[0, 0, 0], [3, 3, 3], [5, 5, 5], [7, 7, 7], [8, 8, 8]]))],  # B/W

        # Annotations style config
        annotations=dict(
            font=dict(
                color="#140054",
                size=18,
                family='Times New Roman'
            ),
            showarrow=False,
            opacity=1.0,
            bgcolor='rgb(255,255,255)'
        )

    )

    config = ad.config.set_default_config(kwargs, config, default_config)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    if repetition_id is None:
        path = experiment_definition['directory']
    else:
        path = os.path.join(
            experiment_definition['directory'],
            'repetition_{:06d}'.format(int(repetition_id))
        )

    path = os.path.join(path, 'statistics')
    final_observation_path = os.path.join(path, 'final_observation')

    # Recover images (from png or zip)
    all_images = []
    if os.path.isdir(final_observation_path):
        dir_content = os.listdir(final_observation_path)
        for image_file_name in dir_content:
            file_path = os.path.join(path, image_file_name)
            if os.path.isfile(file_path) and '.png' in image_file_name:
                item_id = image_file_name.split('.')[0]
                item_id = int(item_id)
                file = open(file_path, "rb")
                image_PIL = Image.open(file)
                if config.images_transform:
                    image_PIL = transform_image_from_colormap(image_PIL, config.images_transform_colormaps[0])
                all_images.append(image_PIL)

    elif zipfile.is_zipfile(final_observation_path + '.zip'):
        zf = zipfile.ZipFile(final_observation_path + '.zip', 'r')
        dir_content = zf.namelist()
        for image_file_name in dir_content:
            item_id = image_file_name.split('.')[0]
            item_id = int(item_id)
            with zf.open(image_file_name) as file:
                image_PIL = Image.open(file)
                if config.images_transform:
                    image_PIL = transform_image_from_colormap(image_PIL, config.images_transform_colormaps[0])
                all_images.append(image_PIL)

    else:
        warnings.warn('No Images found under {!r}'.format(final_observation_path))

    # loads pathes taken in HOLMES
    ids_per_node_path = os.path.join(path, 'ids_per_node')
    ids_per_node = dict(np.load(ids_per_node_path + '.npz'))
    for gs_path, gs_val in ids_per_node.items():
        if len(gs_val.shape) == 0:
            ids_per_node[gs_path] = gs_val.dtype.type(gs_val)

    pathes = []
    max_depth = 0
    for k in ids_per_node.keys():
        cur_node_path = k[3:]
        pathes.append(cur_node_path)
        if (len(cur_node_path) - 1) > max_depth:
            max_depth = len(cur_node_path) - 1

    # load representations per node in HOLMES
    representations_per_node_path = os.path.join(path, 'representations')
    representations_per_node = dict(np.load(representations_per_node_path + '.npz'))
    for gs_path, gs_val in representations_per_node.items():
        representations_per_node[gs_path] = gs_val.dtype.type(gs_val)

    tot_width = config.global_layout.width
    row_height = config.global_layout.height

    figures = []
    counter = 0
    n_total = len(ids_per_node["gs_0"])

    for depth in range(max_depth + 1):
        cur_depth_pathes = []
        for path in pathes:
            if (len(path) - 1) == depth:
                cur_depth_pathes.append(path)
        cur_depth_pathes.sort()

        values = []
        for path in cur_depth_pathes:
            n_cur_node = len(ids_per_node["gs_{}".format(path)])
            values.append(n_cur_node / n_total * tot_width)

        for col_idx in range(len(values)):
            cur_path = cur_depth_pathes[col_idx]
            value = values[col_idx]

            x0 = 0
            y0 = 0
            # v1:
            w = value
            # v2:
            # w = tot_width
            h = row_height

            # images layout
            min_n_cols = 2
            space_between_images = 0.1
            n_cols = int((w - 2 * config.margin_x) // config.images.sizex)
            n_cols = max(n_cols, min_n_cols)
            space_x = config.images.sizex + space_between_images
            centers_x = []
            for j in range(n_cols):
                centers_x.append(j * space_x + space_x / 2.0)

            n_rows = int((h - 2 * config.margin_y) // config.images.sizey)
            space_y = config.images.sizey + space_between_images
            centers_y = []
            for i in range(n_rows):
                centers_y.append(i * space_y + space_y / 2.0)

            cur_path_x_ids = ids_per_node["gs_{}".format(cur_path)]
            indices_diverse = get_diversity_representants(representations_per_node["gs_{}".format(cur_path)], n_candidates=750, n_images=n_rows * n_cols)
            indices_diverse = cur_path_x_ids[indices_diverse]
            #list_of_random_items = np.random.choice(cur_path_x_ids, size=n_rows * n_cols, replace=True)

            images = []
            for i in range(n_rows):
                for j in range(n_cols):
                    image_config = ad.config.set_default_config(
                        dict(
                            source=all_images[
                                indices_diverse[np.ravel_multi_index((i, j), dims=(n_rows, n_cols), order='F')]],
                            x=x0 + config.margin_x + centers_x[j],
                            y=y0 + config.margin_y + centers_y[i],
                        ),
                        config.images)
                    images.append(image_config)


            shape_w = n_cols*config.images.sizex + (n_cols - 1) * space_between_images + 2*config.margin_x
            shape_h = n_rows * config.images.sizey + (n_rows - 1) * space_between_images + 2 * config.margin_y
            annotation_w = 170
            annotation_h = 20
            final_w = max(shape_w, annotation_w)
            final_h = shape_h + annotation_h + 0.2

            # annotations layout
            annotation_config = ad.config.set_default_config(
                dict(
                    x=x0 + (final_w / 2),
                    y=y0 + shape_h + 0.2 + annotation_h,
                    width=annotation_w,
                    height=annotation_h,
                    text="<b>GS {}: {:.1f}%<b>".format(cur_path, w / tot_width * 100),
                    font=dict(
                        color=config.shapes_lines_colors[counter],
                    )
                ),
                config.annotations
            )

            # shapes layout
            shape_config = ad.config.set_default_config(
                dict(
                    type='rect',
                    x0=x0,
                    y0=y0,
                    x1=x0 + shape_w,
                    y1=y0 + shape_h,
                    fillcolor=config.shapes_background_colors[counter],
                    line=dict(color=config.shapes_lines_colors[counter])
                ),
                config.shapes
            )


            # figure
            cur_fig_layout = ad.config.set_default_config(
                dict(
                    width=final_w,
                    height=final_h,
                    margin=dict(autoexpand=True),
                    annotations=[annotation_config],
                    shapes=[shape_config],
                    images=images
                ),
                config.global_layout
            )

            figure = dict(data=[go.Scatter()], layout=cur_fig_layout)
            plotly.offline.iplot(figure)
            figures.append(figure)

            counter = counter + 1
            if counter >= len(config.shapes_background_colors):
                counter = 0

    return figures
