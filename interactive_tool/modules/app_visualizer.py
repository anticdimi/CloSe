import os
from pathlib import Path

import numpy as np
import open3d.visualization as vis
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


from .inference import InferenceWrapper
from .settings import DEFAULT_MATERIAL
from .utils import blend_colors, save_file, to_vector3d, assign_ids, find_save, get_timestamp
from .utils import relabel, load_mesh


class AppWindowO3DVisualizer:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(
        self,
        window_name,
        width,
        height,
        palette,
        pretrained_path,
        **kwargs,
    ) -> None:
        self.app = gui.Application.instance
        self.app.initialize()

        # Initialize visualizer window
        self.vis = vis.O3DVisualizer(window_name, width, height)
        self.vis.show_settings = True

        # Initialize (to None) data path belonging to the current mesh
        self.cwd = os.getcwd()
        self.dataset_path = None
        self.data_path = None
        self.scan_name = None
        self.id_dict = None
        self.scan_no = -1
        self.selected_set = None

        # Initialize palette
        self.color_palette = palette

        # Initialize hyperparams and configurations
        self.alpha = None
        self.texthreshold = kwargs['texthreshold']
        self.neighbour_threshold = kwargs['neighbour_threshold']
        self.other_settings = kwargs

        # Initalize default material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        for k, v in DEFAULT_MATERIAL.items():
            setattr(self.mat, f'base_{k}', v)
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        self.mat.point_size = 3 * self.vis.scaling

        # Initialize inference wrapper
        self.inference = InferenceWrapper(pretrained_path)

        # Initialize mesh
        self.current_mesh, self.current_npz = None, None
        self.selected_ids = set([])

        # Initialize button actions for CloSeT
        self._init_custom_actions()

        self.app.add_window(self.vis)

    def _init_custom_actions(self):
        def _assign_labels(o3dvis):
            """Change the label of the selected vertices to the selected label."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self._update_selected_points()
            selected_vertices = list(self.selected_ids)
            np.save(self.data_path.with_name('temp.npy'), self.current_npz['labels'])
            self.current_mesh = relabel(
                self.current_mesh, o3dvis.selected_label, selected_vertices, self.color_palette
            )
            self.current_npz['labels'][selected_vertices] = o3dvis.selected_label

            blend_colors(
                self.current_mesh,
                self.current_npz,
                self.color_palette,
                selected_vertices,
                self.alpha,
            )
            self._update_geometry(False)

        def _assign_by_voting(o3dvis):
            """Change the label of the selected vertices to the majority label in the selected vertices."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self._update_selected_points()
            selected_vertices = list(self.selected_ids)
            if len(selected_vertices) == 0:
                self.vis.show_message_box('Info', 'No points selected!')
                return
            sel_labels = self.current_npz['labels'][selected_vertices]
            values, counts = np.unique(sel_labels, return_counts=True)
            new_label = values[np.argmax(counts)]
            np.save(self.data_path.with_name('temp.npy'), self.current_npz['labels'])
            self.current_mesh = relabel(
                self.current_mesh,
                new_label,
                selected_vertices,
                self.color_palette,
            )
            self.current_npz['labels'][selected_vertices] = new_label

            blend_colors(
                self.current_mesh,
                self.current_npz,
                self.color_palette,
                selected_vertices,
                self.alpha,
            )
            self._update_geometry(False)

        def _change_all(o3dvis):
            """Change all the vertices, that has same label as the selected vertex, to the selected label."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self._update_selected_points()
            reference_vertices = list(self.selected_ids)
            if len(reference_vertices) != 1:
                self.vis.show_message_box('Info', 'Single point needs to be selected!')
                return
            sel_labels = self.current_npz['labels'][reference_vertices]
            values, counts = np.unique(sel_labels, return_counts=True)
            changed_label = values[np.argmax(counts)]
            # find all indices of the vertices with the changed_label
            changed_vertices = np.where(self.current_npz['labels'] == changed_label)[0]
            np.save(self.data_path.with_name('temp.npy'), self.current_npz['labels'])
            self.current_mesh = relabel(
                self.current_mesh,
                o3dvis.selected_label,
                changed_vertices,
                self.color_palette,
            )
            self.current_npz['labels'][changed_vertices] = o3dvis.selected_label

            blend_colors(
                self.current_mesh,
                self.current_npz,
                self.color_palette,
                changed_vertices,
                self.alpha,
            )
            self._update_geometry(False)

        def _change_neighbors(o3dvis):
            """Change the all the vertices within a threshold distance of the selected vertices to the selected label."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self._update_selected_points()
            reference_vertices = list(self.selected_ids)
            if len(reference_vertices) == 0:
                self.vis.show_message_box('Info', 'Select a single vertex!')
                return
            changed_vertices = np.array([])
            for reference_vertex in reference_vertices:
                changed_label = self.current_npz['labels'][reference_vertex]
                # find all indices of the vertices with the changed_label
                changed_vertices_all = np.where(self.current_npz['labels'] == changed_label)[0]
                ref_point = np.asarray(self.current_mesh.vertices[reference_vertex]).reshape(1, -1)
                # change all vertices within a threshold distance
                point_with_label = np.asarray(self.current_mesh.vertices)[changed_vertices_all]
                distances = np.linalg.norm(point_with_label - ref_point, axis=1)
                changed_vertices = np.append(
                    changed_vertices, [changed_vertices_all[distances < self.neighbour_threshold]]
                )
            changed_vertices = np.unique(changed_vertices).reshape(-1).astype(np.int32)
            np.save(self.data_path.with_name('temp.npy'), self.current_npz['labels'])
            self.current_mesh = relabel(
                self.current_mesh,
                o3dvis.selected_label,
                changed_vertices,
                self.color_palette,
            )
            self.current_npz['labels'][changed_vertices] = o3dvis.selected_label

            blend_colors(
                self.current_mesh,
                self.current_npz,
                self.color_palette,
                changed_vertices,
                self.alpha,
            )
            self._update_geometry(reset_camera=False)

        def _change_by_texture(o3dvis):
            """Change the labels based on the texture similarity."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self._update_selected_points()
            reference_vertices = list(self.selected_ids)

            if len(reference_vertices) == 0:
                self.vis.show_message_box('Info', 'Select a single vertex!')
                return
            changed_vertices = np.array([])
            data_colors = (
                self.current_npz['colors']
                if len(self.current_npz['colors'].shape) == 2
                else self.current_npz['colors'][0, :, :]
            )
            if np.any(data_colors > 1):
                data_colors = data_colors / 255.0

            texture = to_vector3d(data_colors)
            for reference_vertex in reference_vertices:
                ref_texture = texture[reference_vertex]
                # find all indices of the vertices with similar texture color
                condition = np.linalg.norm(texture - ref_texture, axis=1) < self.texthreshold
                changed_vertices_all = np.where(condition)[0]
                ref_point = np.asarray(self.current_mesh.vertices[reference_vertex]).reshape(1, -1)
                # change all vertices within a threshold distance
                point_with_texture = np.asarray(self.current_mesh.vertices)[changed_vertices_all]
                distances = np.linalg.norm(point_with_texture - ref_point, axis=1)
                changed_vertices = np.append(
                    changed_vertices, [changed_vertices_all[distances < self.neighbour_threshold]]
                )
            changed_vertices = np.unique(changed_vertices).reshape(-1).astype(np.int32)
            np.save(self.data_path.with_name('temp.npy'), self.current_npz['labels'])
            self.current_mesh = relabel(
                self.current_mesh,
                o3dvis.selected_label,
                changed_vertices,
                self.color_palette,
            )
            self.current_npz['labels'][changed_vertices] = o3dvis.selected_label

            blend_colors(
                self.current_mesh,
                self.current_npz,
                self.color_palette,
                changed_vertices,
                self.alpha,
            )
            self._update_geometry(reset_camera=False)

        def _undo_last(o3dvis):
            """Undo the last change made to the labels."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            data_labels = find_save(self.data_path, 'temp.npy')
            if data_labels is False:
                self.vis.show_message_box('Info', 'No previous labels found!')
                return
            self.current_npz['labels'] = data_labels

            blend_colors(self.current_mesh, self.current_npz, self.color_palette, alpha=self.alpha)
            self._update_geometry(reset_camera=False)

        def _infer(o3dvis, save_stats=False):
            """Infer the segmentation for the current mesh."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self.vis.show_message_box('Info', 'The segmentation has been started!')
            outp_dict = self.inference.infer(self.data_path)
            unique_labels = np.unique(outp_dict['pred_labels'])
            for label in unique_labels:
                selected_vertices = np.where(outp_dict['pred_labels'] == label)[0]
                changed_vertices = outp_dict['idx'][selected_vertices]
                self.current_mesh = relabel(
                    self.current_mesh,
                    label,
                    changed_vertices,
                    self.color_palette,
                )
                self.current_npz['labels'][changed_vertices] = outp_dict['pred_labels'][
                    selected_vertices
                ]

            blend_colors(self.current_mesh, self.current_npz, self.color_palette, alpha=self.alpha)
            if save_stats:
                # save inside results folder
                results_dir = self.dataset_path / 'results'
                os.makedirs(results_dir, exist_ok=True)
                save_file(results_dir, f'{self.scan_name}_outp_dict.npy', outp_dict)
            self.vis.show_message_box('Info', 'The segmentation is done!')
            self._update_geometry(reset_camera=False)

        def _feedback_loop(o3dvis):
            """This function inputs the mesh pcs as a whole and performs refinement for several steps."""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self.vis.show_message_box('Info', 'The model refinement has been started!')
            selected_labels = self.current_npz['labels']
            selected_ids = np.arange(len(selected_labels)).astype(np.int32)
            if len(selected_ids) > 15000:
                batched_backprop = True
            else:
                batched_backprop = False
            scan_name = Path(self.data_path).stem
            # load the diff npy file
            saved_labels = find_save(self.data_path, f'*{scan_name}.npy')
            based_on_diff = False
            if saved_labels is not False and based_on_diff is True:
                data_labels = self.current_npz['labels']
                changed_ids = np.where(np.abs(saved_labels - data_labels))[0]
                if not len(changed_ids) > 0:
                    changed_ids = []
                self.inference.refine_weights(
                    self.data_path,
                    selected_ids,
                    selected_labels,
                    batched_backprop=batched_backprop,
                    changed_indices=changed_ids,
                )
            else:
                self.inference.refine_weights(
                    self.data_path,
                    selected_ids,
                    selected_labels,
                    batched_backprop=batched_backprop,
                )
            self.vis.show_message_box('Info', 'The model is updated!')

        def _run_evaluation(o3dvis, split='val'):
            self.vis.show_message_box('Info', f'The evaluation on {split} split is started!')
            eval_dict, out_dict = self.inference.evaluate(split=split, palette=self.color_palette)
            message = f'Evaluation mIoU: {eval_dict["mIoU"]:.4f}'
            self.vis.show_message_box('Evaluation Results', message)

        def _save_labels(o3dvis, save_stats=True) -> None:
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            save_path = save_file(self.dataset_path, self.scan_name, self.current_npz['labels'])
            self.vis.show_message_box('Info', f'Saved new labels at {save_path}!')

        def _save_model(o3dvis) -> None:
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            timestamp = get_timestamp()
            save_dir = Path('./models')
            save_dir.mkdir(exist_ok=True, parents=True)

            filename = f'{timestamp}_{self.inference.model_name}.pt'
            print(f'Saving new checkpoint at {save_dir / filename}!')
            self.vis.show_message_box('Info', f'Saving new checkpoint at {save_dir / filename}!')
            self.inference.save_model(save_dir, filename)

        def _load_labels(o3dvis):
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            data_labels = find_save(self.data_path, f'*{self.scan_name}.npy')
            if data_labels is False:
                self.vis.show_message_box('Info', 'No previous labels found!')
                return
            self.current_npz['labels'] = data_labels
            blend_colors(self.current_mesh, self.current_npz, self.color_palette, alpha=self.alpha)
            self._update_geometry(reset_camera=False)

        def _next(o3dvis):
            """Browse to the next mesh in the dataset."""
            if self.scan_no == -1:
                self.vis.show_message_box('Info', 'No data folder loaded!')
                return
            if self.scan_no < len(self.id_dict) - 1:
                self._remove_geometry(removed_scan_no=self.scan_no)
                self.scan_no += 1

                # Assign data path belonging to the next mesh
                self.data_path = Path(f'{self.id_dict[self.scan_no]}')
                self.scan_name = self.data_path.stem
                # Initalize mesh
                self.current_mesh, self.current_npz = load_mesh(
                    self.data_path, self.color_palette, alpha=self.alpha
                )
                self._add_geometry()

        def _prev(o3dvis):
            """Browse to the previous mesh in the dataset."""
            if self.scan_no == -1:
                self.vis.show_message_box('Info', 'No data folder loaded!')
                return
            if self.scan_no > 0:
                self._remove_geometry(removed_scan_no=self.scan_no)
                self.scan_no -= 1

                # Assign data path belonging to the next mesh
                self.data_path = Path(f'{self.id_dict[self.scan_no]}')
                self.scan_name = self.data_path.stem
                # Initalize mesh
                self.current_mesh, self.current_npz = load_mesh(
                    self.data_path, self.color_palette, alpha=self.alpha
                )
                self._add_geometry()

        def _init_mesh(o3dvis):
            """Load the mesh and the corresponding npz file"""
            # if not ending with .npz, return
            if o3dvis.selected_path is None or not o3dvis.selected_path[-4:] == '.npz':
                self.vis.show_message_box('Info', 'Select a valid path!')
                return
            self._remove_geometry()
            # Assign the dataset path and the scan name
            self.dataset_path = Path(o3dvis.selected_path).parent
            self.scan_name = Path(o3dvis.selected_path).stem
            self.id_dict, self.scan_no = assign_ids(self.dataset_path, o3dvis.selected_path)
            self.data_path = Path(self.id_dict[self.scan_no])
            self.alpha = o3dvis.slider_value
            self.current_mesh, self.current_npz = load_mesh(
                self.vis.selected_path, self.color_palette, alpha=self.alpha
            )
            os.chdir(self.cwd)
            # Refresh the visualizer
            self._add_geometry()

        def _change_alpha(o3dvis):
            """Change the alpha value for the mesh visualization"""
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            self.alpha = o3dvis.slider_value
            if self.current_mesh is None:
                self.vis.show_message_box('Info', 'No mesh loaded!')
                return
            blend_colors(self.current_mesh, self.current_npz, self.color_palette, alpha=self.alpha)
            self._update_geometry(False)

        self.vis.path_add_action('           Load Mesh              ', _init_mesh)
        self.vis.labelling_add_action(
            '            Relabel (Majority Vote)           ', _assign_by_voting
        )
        self.vis.labelling_add_action(
            '            Relabel (User Selection)          ', _assign_labels
        )
        self.vis.labelling_add_action('Change neighbors', _change_neighbors)
        self.vis.labelling_add_action('     Change all     ', _change_all)
        self.vis.labelling_add_action('Change by texture', _change_by_texture)
        self.vis.labelling_add_action('      Undo last      ', _undo_last)
        self.vis.labelling_add_action('     Save labels     ', _save_labels)
        self.vis.labelling_add_action('     Load labels     ', _load_labels)
        self.vis.inference_add_action('         Predict         ', _infer)
        self.vis.inference_add_action('        Evaluate         ', _run_evaluation)
        self.vis.model_add_action('    Refine model    ', _feedback_loop)
        self.vis.model_add_action('     Save model     ', _save_model)
        self.vis.browsing_add_action('       Previous        ', _prev)
        self.vis.browsing_add_action('            Next            ', _next)
        self.vis.labelling_control_add_action('     Apply    ', _change_alpha)

    def _add_geometry(self, reset_camera=True, scan_name=None, current_mesh=None):
        if scan_name is not None and current_mesh is not None:
            self.vis.add_geometry(scan_name, current_mesh, self.mat)
        else:
            self.vis.add_geometry(self.scan_name, self.current_mesh, self.mat)
        if reset_camera:
            self.vis.reset_camera_to_default()

    def _remove_geometry(self, removed_scan_no=None, removed_scan_name=None, just_update=False):
        """Remove the geometry in the visualizer."""
        if just_update:
            if removed_scan_no is not None:
                removed_scan_name = Path(self.id_dict[removed_scan_no]).stem
                self.vis.remove_geometry(removed_scan_name)
            elif removed_scan_name is not None:
                self.vis.remove_geometry(removed_scan_name)
            elif self.scan_name is not None:
                self.vis.remove_geometry(self.scan_name)
        else:
            if removed_scan_no is not None:
                removed_scan_name = Path(self.id_dict[removed_scan_no]).stem
                self.vis.remove_geometry_and_data(removed_scan_name)
            elif removed_scan_name is not None:
                self.vis.remove_geometry_and_data(removed_scan_name)
            elif self.scan_name is not None:
                self.vis.remove_geometry_and_data(self.scan_name)
        self._unselect_all_points()

    def _update_geometry(self, reset_camera=True, removed_scan_no=None, removed_scan_name=None):
        """Update the geometry in the visualizer."""
        self._remove_geometry(removed_scan_no, removed_scan_name, just_update=False)
        self._add_geometry(reset_camera)

    def _update_selected_points(self):
        """Update the selected vertices."""
        selection_sets = self.vis.get_selection_sets()
        if len(selection_sets) > 0:
            selected_pc_set = selection_sets[self.vis.selected_set]
            if not len(selected_pc_set) > 0:
                self.vis.show_message_box('Info', 'No points selected for the set!')
                return
            self.selected_set = self.vis.selected_set
            selected_object_pc = selection_sets[self.selected_set][self.scan_name]
            picked = sorted(list(selected_object_pc), key=lambda x: x.order)
            indices = [idx.index for idx in picked]
            self.selected_ids.update(indices)
        else:
            self.vis.show_message_box('Info', 'No points selected!')

    def _unselect_all_points(self):
        self.selected_ids = set([])
        self.selected_set = None

    def run(self):
        self.app.run()
