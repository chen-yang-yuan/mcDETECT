import math
import miniball
import numpy as np
import pandas as pd
from rtree import index
from scipy.spatial import cKDTree
from scipy.stats import poisson
from shapely.geometry import Point
from sklearn.cluster import DBSCAN


def make_tree(d1 = None, d2 = None, d3 = None):
    active_dimensions = [dimension for dimension in [d1, d2, d3] if dimension is not None]
    if len(active_dimensions) == 1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions) == 2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    elif len(active_dimensions) == 3:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    return cKDTree(points)


def make_rtree(spheres):
    p = index.Property()
    idx = index.Index(properties = p)
    for i, sphere in enumerate(spheres.itertuples()):
        center = Point(sphere.sphere_x, sphere.sphere_y)
        bounds = (
            center.x - sphere.sphere_r,
            center.y - sphere.sphere_r,
            center.x + sphere.sphere_r,
            center.y + sphere.sphere_r
        )
        idx.insert(i, bounds)
    return idx


class model:
    """
    Optimized version:
      - Builds ONE global KD-tree for all transcripts (self._tree_all)
      - Converts DataFrame columns to NumPy once and reuses them
      - Avoids per-marker tree rebuilds and heavy Pandas ops in inner loops
      - Uses configurable miniball epsilon (default 1e-4) for speed
    """

    def __init__(
        self,
        shape,
        transcripts: pd.DataFrame,
        target_all,
        eps=1.5,
        cutoff=0.95,
        alpha=5,
        low=3,
        in_thr=0.5,
        comp_thr=2,
        size_thr=4,
        l=1,
        p=0.2,
        s=1,
        miniball_epsilon=1e-4,   # <- looser tolerance for speed
    ):
        self.shape = shape
        self.transcripts = transcripts.copy()
        self.target_all = list(target_all)
        self.eps = eps
        self.cutoff = cutoff
        self.alpha = alpha
        self.low = low
        self.in_thr = in_thr
        self.comp_thr = comp_thr
        self.size_thr = size_thr
        self.l = l
        self.p = p
        self.s = s
        self.miniball_epsilon = miniball_epsilon

        # Precompute area once
        self._effect_area = float(self.shape[0]) * float(self.shape[1])

        # Ensure 'target' is a categorical aligned with target_all for fast integer coding
        if not pd.api.types.is_categorical_dtype(self.transcripts['target']):
            self.transcripts['target'] = pd.Categorical(
                self.transcripts['target'], categories=self.target_all, ordered=False
            )
        else:
            # align categories to target_all order if needed
            if list(self.transcripts['target'].cat.categories) != list(self.target_all):
                self.transcripts['target'] = self.transcripts['target'].cat.set_categories(self.target_all, ordered=False)

        # Precompute NumPy views for speed
        self._x = self.transcripts['global_x'].to_numpy(dtype=float, copy=False)
        self._y = self.transcripts['global_y'].to_numpy(dtype=float, copy=False)
        self._z = self.transcripts['global_z'].to_numpy(dtype=float, copy=False)
        self._coords = np.column_stack((self._x, self._y, self._z))
        self._in_nucleus = self.transcripts['in_nucleus'].to_numpy(dtype=np.int8, copy=False)

        # Integer codes for targets (0..K-1); fast comparisons & unique()
        self._target_codes = self.transcripts['target'].cat.codes.to_numpy(copy=False)
        # Map marker string -> int code
        self._code_for = {name: i for i, name in enumerate(self.target_all)}

        # Global KD-tree over all transcripts (used for "others" queries)
        self._tree_all = make_tree(d1=self._x, d2=self._y, d3=self._z)

        # Precompute counts per marker for poisson_select()
        self._counts_by_code = np.bincount(
            np.clip(self._target_codes, 0, len(self.target_all) - 1),
            minlength=len(self.target_all)
        )

    # ------------------------ Utilities ------------------------

    def effect_area(self):
        # Kept for API compatibility; already cached in self._effect_area
        return self._effect_area

    def poisson_select(self, gene_name):
        code = self._code_for[gene_name]
        num_trans = int(self._counts_by_code[code])
        bg_density = num_trans / self._effect_area
        # Expected #points in eps-ball scaled by alpha, then take Poisson cutoff quantile
        cutoff_density = poisson.ppf(self.cutoff, mu=self.alpha * bg_density * (np.pi * self.eps ** 2))
        optimal_m = int(max(cutoff_density, self.low))
        return optimal_m

    # ------------------------ Single target DBSCAN ------------------------

    def dbscan_single(self, target_name, print_itr=False):
        code = self._code_for[target_name]
        mask = (self._target_codes == code)

        # Fast paths to arrays for this target
        coords_t = self._coords[mask]
        in_t = self._in_nucleus[mask]

        if coords_t.shape[0] == 0:
            if print_itr:
                print(f'Gene: {target_name}; Transcripts: 0; Min_samples: N/A; Filtered aggregations: 0')
            return pd.DataFrame(columns=['sphere_x', 'sphere_y', 'sphere_z', 'sphere_r', 'gene'])

        min_spl = self.poisson_select(target_name)

        # 3D DBSCAN on this target's coords
        db = DBSCAN(eps=self.eps, min_samples=min_spl, algorithm='kd_tree').fit(coords_t)
        labels = db.labels_
        # Identify cluster ids (exclude -1 noise)
        unique_labels = np.unique(labels)
        cluster_ids = unique_labels[unique_labels != -1]

        sphere_x, sphere_y, sphere_z, sphere_r = [], [], [], []

        for k in cluster_ids:
            k_mask = (labels == k)
            temp_coords = coords_t[k_mask]
            temp_size = temp_coords.shape[0]
            if temp_size == 0:
                continue

            in_count = int(in_t[k_mask].sum())
            center, r2 = miniball.get_bounding_ball(temp_coords, epsilon=self.miniball_epsilon)
            r = math.sqrt(r2)
            in_ratio = in_count / float(temp_size)

            if (in_ratio <= self.in_thr) and (r <= self.size_thr):
                sphere_x.append(center[0])
                sphere_y.append(center[1])
                sphere_z.append(center[2])
                sphere_r.append(r)

        sphere = pd.DataFrame({
            'sphere_x': sphere_x,
            'sphere_y': sphere_y,
            'sphere_z': sphere_z,
            'sphere_r': sphere_r,
            'gene': [target_name] * len(sphere_x)
        })

        if print_itr:
            print(f'Gene: {target_name}; Transcripts: {coords_t.shape[0]}; Min_samples: {min_spl}; Filtered aggregations: {sphere.shape[0]}')

        return sphere

    # ------------------------ Multi-target DBSCAN with composition filter ------------------------

    def dbscan(self, target_names=None, print_itr=False):
        if target_names is None:
            target_names = self.target_all

        result_data = {}

        for j in target_names:
            code_j = self._code_for[j]
            mask_j = (self._target_codes == code_j)

            coords_t = self._coords[mask_j]
            in_t = self._in_nucleus[mask_j]

            if coords_t.shape[0] == 0:
                # Return empty frame for this marker index to preserve structure
                sphere = pd.DataFrame(columns=['sphere_x', 'sphere_y', 'sphere_z', 'sphere_r', 'size', 'comp', 'gene'])
                result_data[self.target_all.index(j)] = sphere
                if print_itr:
                    print(f'Gene: {j}; Transcripts: 0; Min_samples: N/A; Filtered aggregations: 0')
                continue

            min_spl = self.poisson_select(j)
            db = DBSCAN(eps=self.eps, min_samples=min_spl, algorithm='kd_tree').fit(coords_t)
            labels = db.labels_
            unique_labels = np.unique(labels)
            cluster_ids = unique_labels[unique_labels != -1]

            sphere_x, sphere_y, sphere_z, sphere_r, sphere_size, sphere_comp = [], [], [], [], [], []

            for k in cluster_ids:
                k_mask = (labels == k)
                temp_coords = coords_t[k_mask]
                temp_size = temp_coords.shape[0]
                if temp_size < self.comp_thr:  # tiny clusters unlikely to pass comp filter
                    continue

                in_count = int(in_t[k_mask].sum())
                center, r2 = miniball.get_bounding_ball(temp_coords, epsilon=self.miniball_epsilon)
                r = math.sqrt(r2)
                in_ratio = in_count / float(temp_size)

                if (in_ratio > self.in_thr) or (r > self.size_thr):
                    continue

                # Composition filter: query neighbors in the global tree, then exclude same-marker points
                idx_other = self._tree_all.query_ball_point([center[0], center[1], center[2]], r)
                if len(idx_other) == 0:
                    total_comp = 1
                    other_size = 0
                else:
                    # Exclude same marker points by code
                    idx_other = np.asarray(idx_other, dtype=np.int64)
                    other_codes = self._target_codes[idx_other]
                    other_size = int((other_codes != code_j).sum())
                    # Unique number of OTHER marker types
                    other_comp = int(np.unique(other_codes[other_codes != code_j]).size)
                    total_comp = 1 + other_comp

                if total_comp >= self.comp_thr:
                    sphere_x.append(center[0])
                    sphere_y.append(center[1])
                    sphere_z.append(center[2])
                    sphere_r.append(r)
                    sphere_size.append(int(temp_size + other_size))
                    sphere_comp.append(int(total_comp))

            sphere = pd.DataFrame({
                'sphere_x': sphere_x,
                'sphere_y': sphere_y,
                'sphere_z': sphere_z,
                'sphere_r': sphere_r,
                'size': sphere_size,
                'comp': sphere_comp,
                'gene': [j] * len(sphere_x)
            })
            result_data[self.target_all.index(j)] = sphere

            if print_itr:
                print(f'Gene: {j}; Transcripts: {coords_t.shape[0]}; Min_samples: {min_spl}; Filtered aggregations: {sphere.shape[0]}')

        return result_data

    # ------------------------ Utilities for merging ------------------------

    def find_points(self, sphere_a: pd.Series, sphere_b: pd.Series):
        """
        Returns union of points from gene A and gene B within their respective spheres (global coords).
        Uses prebuilt global tree; filters by target code instead of string for speed.
        """
        # Query A
        idx_a = self._tree_all.query_ball_point(
            [sphere_a['sphere_x'], sphere_a['sphere_y'], sphere_a['sphere_z']],
            sphere_a['sphere_r']
        )
        idx_a = np.asarray(idx_a, dtype=np.int64)
        code_a = self._code_for[sphere_a['gene']]
        mask_a = (self._target_codes[idx_a] == code_a)
        pts_a = self._coords[idx_a][mask_a]

        # Query B
        idx_b = self._tree_all.query_ball_point(
            [sphere_b['sphere_x'], sphere_b['sphere_y'], sphere_b['sphere_z']],
            sphere_b['sphere_r']
        )
        idx_b = np.asarray(idx_b, dtype=np.int64)
        code_b = self._code_for[sphere_b['gene']]
        mask_b = (self._target_codes[idx_b] == code_b)
        pts_b = self._coords[idx_b][mask_b]

        if pts_a.size == 0 and pts_b.size == 0:
            return pd.DataFrame(columns=['global_x', 'global_y', 'global_z'])

        pts = np.vstack([pts for pts in (pts_a, pts_b) if pts.size > 0])
        return pd.DataFrame(pts, columns=['global_x', 'global_y', 'global_z'])

    def remove_overlaps(self, set_a: pd.DataFrame, set_b: pd.DataFrame):
        """
        Remove overlapping spheres between two sets based on distance criteria.
        Parameter p controls the threshold for merging: when two spheres overlap and
        dist < p * l * (radius_a + radius_b), they are merged.
        """
        set_a = set_a.copy()
        set_b = set_b.copy()

        idx_b = make_rtree(set_b)
        for i, sphere_a in set_a.iterrows():
            bounds_a = (
                sphere_a.sphere_x - sphere_a.sphere_r,
                sphere_a.sphere_y - sphere_a.sphere_r,
                sphere_a.sphere_x + sphere_a.sphere_r,
                sphere_a.sphere_y + sphere_a.sphere_r
            )
            possible_overlaps = idx_b.intersection(bounds_a)

            center_a_3D = (sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z)

            for j in possible_overlaps:
                if j in set_b.index:
                    sphere_b = set_b.loc[j]
                    center_b_3D = (sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z)
                    dist = math.dist(center_a_3D, center_b_3D)
                    radius_sum = sphere_a.sphere_r + sphere_b.sphere_r
                    radius_diff = sphere_a.sphere_r - sphere_b.sphere_r

                    c0 = (dist < self.l * radius_sum)
                    c1 = (dist <= self.l * abs(radius_diff))
                    c1_1 = (radius_diff > 0)
                    c2_1 = (dist < self.p * self.l * radius_sum)

                    if c0:
                        if c1 and c1_1:
                            set_b.drop(index=j, inplace=True)
                        elif c1 and (not c1_1):
                            set_a.loc[i] = set_b.loc[j]
                            set_b.drop(index=j, inplace=True)
                        elif (not c1) and c2_1:
                            points_union = np.array(self.find_points(sphere_a, sphere_b))
                            if points_union.size == 0:
                                set_b.drop(index=j, inplace=True)
                                continue
                            new_center, new_radius = miniball.get_bounding_ball(points_union, epsilon=self.miniball_epsilon)
                            set_a.loc[i, 'sphere_x'] = new_center[0]
                            set_a.loc[i, 'sphere_y'] = new_center[1]
                            set_a.loc[i, 'sphere_z'] = new_center[2]
                            set_a.loc[i, 'sphere_r'] = self.s * new_radius
                            set_b.drop(index=j, inplace=True)

        set_a = set_a.reset_index(drop=True)
        set_b = set_b.reset_index(drop=True)
        return set_a, set_b

    def merge_sphere(self, sphere_dict, print_itr=False):
        # Concatenate spheres marker-by-marker, merging overlaps progressively
        sphere = sphere_dict[0].copy()
        for j in range(1, len(self.target_all)):
            target_sphere = sphere_dict[j]
            sphere, target_sphere_new = self.remove_overlaps(sphere, target_sphere)
            if print_itr:
                print(sphere.shape[0], target_sphere.shape[0])
            sphere = pd.concat([sphere, target_sphere_new], ignore_index=True)
            if print_itr:
                print(sphere.shape[0])
        return sphere

    def merge_data(self, print_dbscan=False, print_merge=False):
        result_data = self.dbscan(print_itr=print_dbscan)
        sphere = self.merge_sphere(result_data, print_itr=print_merge)
        return sphere
