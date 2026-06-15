# Tessellated RVE geometry builder: labels grain-boundary core/slide segments,
# detects periodic pairs, and returns NGSolve meshes with ContactBoundary metadata.


import numpy as np
from collections import defaultdict
from netgen.occ import WorkPlane, Compound, OCCGeometry
from ngsolve import *


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def approx(u, v, tol):
    """Return True if |u - v| ≤ tol."""
    return abs(u - v) <= tol


def edge_keys_for_region(reg):
    """
    Given a polygon defined by a list/tuple of 1-based vertex indices (CCW),
    return:

        unordered_edges: list of (min_idx, max_idx)
        directed_edges:  list of (i_k, i_{k+1})

    The last edge closes the polygon.
    """
    n = len(reg)
    directed = [(reg[k], reg[(k + 1) % n]) for k in range(n)]
    unordered = [tuple(sorted(pair)) for pair in directed]
    return unordered, directed


def build_shared_edge_map(regions):
    """
    Identify ownership of polygon edges across all regions.

    Returns
    -------
    shared : dict
        Maps unordered edge key (min_idx, max_idx) → list of tuples
        [(region_index, directed_edge)] for the two owning regions.
    boundary : dict
        Maps unordered edge key → [(region_index, directed_edge)] for edges
        that are owned by a single region (i.e. external boundary edges).
    """
    owners = {}
    for r_idx, reg in enumerate(regions, start=1):
        unordered, directed = edge_keys_for_region(reg)
        for ukey, dpair in zip(unordered, directed):
            owners.setdefault(ukey, []).append((r_idx, dpair))

    shared = {k: v for k, v in owners.items() if len(v) == 2}
    boundary = {k: v for k, v in owners.items() if len(v) == 1}
    return shared, boundary


def classify_external_edges(points, boundary_map, tol=1e-8):
    """
    Pair single-owner edges into periodic counterparts and treat every match
    as a generic (potentially diagonal) outer boundary pairing.

    Parameters
    ----------
    points : list[(float, float)]
        Global vertex coordinates.
    boundary_map : dict
        Output from build_shared_edge_map containing edges with a single owner.
    tol : float
        Tolerance used when matching edges and translations.

    Returns
    -------
    edge_labels : dict
        Populated for backward compatibility but no longer used for
        directional tagging.
    corner_vertices : dict
        Corner metadata (currently empty placeholder for compatibility).
    external_metadata : dict
        Maps edge keys to dictionaries with pairing data (pair_id, side,
        displacement, prefix) for outer contact construction.
    outer_pairs : list
        Each entry summarises a periodic pairing with keys
        ('pair_id', 'displacement', 'edges'). The displacement vector points
        from the "minus" side towards the "plus" side.
    outer_kink_vertices : set
        Vertex indices on the outer boundary that should be treated like
        "triple junctions" when splitting into core/slide segments.
    """
    if not boundary_map:
        return {}, {}, {}, [], set()

    edge_infos = []
    for owners in boundary_map.values():
        r_idx, (v0, v1) = owners[0]
        p0 = np.array(points[v0 - 1], dtype=float)
        p1 = np.array(points[v1 - 1], dtype=float)
        vec = p1 - p0
        length = np.linalg.norm(vec)
        mid = 0.5 * (p0 + p1)
        edge_infos.append({
            "key": (r_idx, v0, v1),
            "region": r_idx,
            "start": v0,
            "end": v1,
            "p0": p0,
            "p1": p1,
            "vec": vec,
            "length": length,
            "mid": mid,
        })

    if len(edge_infos) % 2:
        raise ValueError("External boundary edges must come in pairs for periodicity.")

    len_tol_scale = max(tol * 10.0, 1e-10)
    trans_tol = max(tol * 10.0, 1e-9)

    edge_labels = {}
    external_metadata = {}
    outer_pairs = []
    pair_counter = 0
    used = [False] * len(edge_infos)

    def translation_between(edge_a, edge_b):
        delta0 = edge_b["p0"] - edge_a["p0"]
        delta1 = edge_b["p1"] - edge_a["p1"]
        if np.linalg.norm(delta0 - delta1) <= trans_tol:
            return 0.5 * (delta0 + delta1)
        delta0 = edge_b["p0"] - edge_a["p1"]
        delta1 = edge_b["p1"] - edge_a["p0"]
        if np.linalg.norm(delta0 - delta1) <= trans_tol:
            return 0.5 * (delta0 + delta1)
        return None

    for i, info_i in enumerate(edge_infos):
        if used[i]:
            continue

        match_idx = None
        match_translation = None
        fallback_idx = None
        fallback_translation = None

        vec = info_i["vec"]
        orientation_is_horizontal = abs(vec[0]) >= abs(vec[1])

        for j in range(i + 1, len(edge_infos)):
            if used[j]:
                continue

            info_j = edge_infos[j]

            max_len = max(info_i["length"], info_j["length"], 1.0)
            if abs(info_i["length"] - info_j["length"]) > len_tol_scale * max_len:
                continue

            translation = translation_between(info_i, info_j)
            if translation is None:
                continue

            prefer_vertical_shift = abs(translation[1]) >= abs(translation[0])
            prefer = prefer_vertical_shift if orientation_is_horizontal else not prefer_vertical_shift

            if prefer:
                match_idx = j
                match_translation = translation
                break

            if fallback_idx is None:
                fallback_idx = j
                fallback_translation = translation

        if match_idx is None:
            if fallback_idx is None:
                raise ValueError("Could not find periodic partner for boundary edge.")
            match_idx = fallback_idx
            match_translation = fallback_translation

        used[i] = True
        used[match_idx] = True
        info_j = edge_infos[match_idx]

        pair_counter += 1

        minus_info, plus_info = sorted([info_i, info_j], key=lambda entry: (entry["mid"][0], entry["mid"][1]))

        translation_vec = np.array(match_translation, dtype=float)
        if minus_info is info_i and plus_info is info_j:
            displacement = translation_vec
        elif minus_info is info_j and plus_info is info_i:
            displacement = -translation_vec
        else:
            # Fallback: align via midpoint difference if identities were lost.
            displacement = np.array(plus_info["mid"], dtype=float) - np.array(minus_info["mid"], dtype=float)
        displacement_tuple = (float(displacement[0]), float(displacement[1]))

        minus_prefix = f"outer_{pair_counter}_minus"
        plus_prefix = f"outer_{pair_counter}_plus"

        edge_labels[minus_info["key"]] = "outer"
        edge_labels[plus_info["key"]] = "outer"

        external_metadata[minus_info["key"]] = {
            "pair_id": pair_counter,
            "side": "minus",
            "base_label": "outer",
            "prefix": minus_prefix,
            "displacement": displacement_tuple,
            "core_labels": {},
            "slide_labels": {},
        }
        external_metadata[plus_info["key"]] = {
            "pair_id": pair_counter,
            "side": "plus",
            "base_label": "outer",
            "prefix": plus_prefix,
            "displacement": displacement_tuple,
            "core_labels": {},
            "slide_labels": {},
        }

        def edge_payload(info, prefix, side):
            return {
                "key": info["key"],
                "side": side,
                "prefix": prefix,
            }

        outer_pairs.append({
            "pair_id": pair_counter,
            "displacement": displacement_tuple,
            "edges": [
                edge_payload(minus_info, minus_prefix, "minus"),
                edge_payload(plus_info, plus_prefix, "plus"),
            ],
        })

    vertex_boundaries = {idx: set() for idx in range(1, len(points) + 1)}
    for (reg_idx, v0, v1), label in edge_labels.items():
        vertex_boundaries[v0].add(label)
        vertex_boundaries[v1].add(label)

    corner_vertices = {}

    def select_corner(required_labels, name, key_func):
        candidates = [vid for vid, labs in vertex_boundaries.items()
                      if required_labels.issubset(labs)]
        if not candidates:
            return
        chosen = min(candidates, key=key_func)
        corner_vertices[chosen] = name

    # Corner labelling is deferred to downstream logic; no dedicated LB tag here.

    outer_kink_vertices = set()
    for (reg_idx, v0, v1), meta in external_metadata.items():
        if isinstance(meta, dict):
            outer_kink_vertices.add(v0)
            outer_kink_vertices.add(v1)

    # Do not double-count corner vertices as outer kinks; the dedicated corner
    # segments already handle those.
    outer_kink_vertices.difference_update(corner_vertices.keys())

    return edge_labels, corner_vertices, external_metadata, outer_pairs, outer_kink_vertices


# -----------------------------------------------------------------------------
# Geometry construction: labeling each grain face
# -----------------------------------------------------------------------------
def face_from_region(points, reg, r_idx, shared_map, xmax, ymax,
                     tol=1e-8, triple_pts=None, core_frac=None,
                     corner_core_frac=None, corner_vertices=None,
                     external_labels=None, outer_kink_vertices=None,
                     outer_core_label_map=None, outer_core_sequence=None,
                     outer_core_vertices=None, junction_incidence=None):
    """
    Create a Netgen OCC WorkPlane face for a single region (grain).

    Each edge of the polygon is examined and labeled based on its type:

      - Shared edges between regions:
          * If no triple junction at either end → one 'slide' segment.
          * If triple junction at one end      → 'core' near that end, 'slide' elsewhere.
          * If triple junctions at both ends   → 'core-slide-core' (three segments).

            - Non-shared edges on the outer box:
                    * Labeled as 'left', 'right', 'bottom', or 'top' as appropriate.
                    * If the edge touches the designated corner (LB), a small segment
                        near the corner is split off and labeled with the corner name.

    Parameters
    ----------
    points : list[(float, float)]
        Global vertex coordinates.
    reg : list[int]
        1-based vertex indices defining this region's polygon.
    r_idx : int
        Region index (1-based).
    shared_map : dict
        Output of build_shared_edge_map(), giving shared edges between regions.
    xmax, ymax : float
        Bounding box extents for outer boundary detection.
    tol : float
        Tolerance for coordinate comparisons.
    triple_pts : list[(float, float)] or None
        Coordinates of vertices where ≥3 grains meet (triple junctions).
    core_frac : float or None
        Fraction of internal edge length to label as "core" near a triple
        junction. If None, defaults to maxh (set in caller).
    corner_core_frac : float or None
        Fraction of outer boundary edge near the LB corner to be labeled with
        the corner tag. If None, defaults to core_frac.
    corner_vertices : dict or None
        Maps either vertex indices or coordinate tuples to the LB corner
        label. Vertex-index maps are preferred. Coordinate maps are kept for
        backward compatibility.
    external_labels : dict or None
        Maps (region_idx, start_vertex, end_vertex) → metadata for external
        edges. The metadata can be either a bare string (legacy) or a dict with
        keys 'pair_id', 'side', 'orientation', 'base_label', 'prefix'. When
        None, the routine falls back to bounding-box detection using xmax/ymax.
    outer_kink_vertices : set or None
        Vertex indices on outer edges that should receive 'core' segments in
        addition to interior triple junctions.

    Returns
    -------
    face : OCC face object for this region.
    """
    poly = [points[j - 1] for j in reg]
    wp = WorkPlane()
    wp.MoveTo(*poly[0])
    n = len(poly)

    def near_triple(pt):
        """Check if a point lies near any triple junction."""
        if triple_pts is None:
            return False
        return any(abs(pt[0] - tp[0]) <= tol and abs(pt[1] - tp[1]) <= tol
                   for tp in triple_pts)

    def triple_id(px, py):
        """Return the index of the triple junction at (px, py), or None."""
        if triple_pts is None:
            return None
        for idx, tp in enumerate(triple_pts):
            if abs(px - tp[0]) <= tol and abs(py - tp[1]) <= tol:
                return idx
        return None

    corner_by_vertex = None
    corner_by_coord = None
    if corner_vertices:
        first_key = next(iter(corner_vertices.keys()))
        if isinstance(first_key, int):
            corner_by_vertex = corner_vertices
        else:
            corner_by_coord = corner_vertices

    def corner_label(pt, vid):
        """Return corner label ('LB', 'LT', 'RB') if endpoint is a core corner."""
        if corner_by_vertex is not None:
            return corner_by_vertex.get(vid)
        if corner_by_coord is not None:
            for (cx, cy), lab in corner_by_coord.items():
                if abs(pt[0] - cx) <= tol and abs(pt[1] - cy) <= tol:
                    return lab
        return None

    def register_outer_core(prefix, name, vertex_id=None):
        if outer_core_label_map is not None and prefix:
            if name not in outer_core_label_map[prefix]:
                outer_core_label_map[prefix].add(name)
                if outer_core_sequence is not None:
                    outer_core_sequence.append(name)
        if outer_core_vertices is not None and vertex_id is not None and name:
            outer_core_vertices[vertex_id].add(name)

    def outer_core_name(prefix, point, other_point, vertex_id=None):
        px, py = point
        ox, oy = other_point
        dx = px - ox
        dy = py - oy
        tol_cmp = 1e-12
        if dx > tol_cmp:
            suffix = "upper"
        elif dx < -tol_cmp:
            suffix = "lower"
        elif dy > tol_cmp:
            suffix = "upper"
        else:
            suffix = "lower"
        name = f"core_{prefix}_{suffix}"
        register_outer_core(prefix, name, vertex_id)
        return name

    for k in range(n):
        (x0, y0) = poly[k]
        (x1, y1) = poly[(k + 1) % n]
        ukey = tuple(sorted((reg[k], reg[(k + 1) % n])))
        owners = shared_map.get(ukey)

        # ------------------------------------------------------------------
        # CASE 1: Internal shared edge between two regions (grain boundary)
        # ------------------------------------------------------------------
        if owners is not None and len(owners) == 2:
            other = owners[0][0] if owners[1][0] == r_idx else owners[1][0]
            i, j = (r_idx, other)
            ii, jj = (i, j) if i < j else (j, i)
            lr = "left" if r_idx == ii else "right"

            is_t0, is_t1 = near_triple((x0, y0)), near_triple((x1, y1))
            dx, dy = (x1 - x0), (y1 - y0)

            # Build list of edge segments to draw, each with a "core"/"slide" tag
            # and (for core stubs) the id of the triple junction it abuts.
            segs = []
            if not is_t0 and not is_t1:
                segs.append((x1, y1, "slide", None))
            elif is_t0 and not is_t1:
                xm, ym = x0 + core_frac * dx, y0 + core_frac * dy
                segs += [(xm, ym, "core", triple_id(x0, y0)), (x1, y1, "slide", None)]
            elif not is_t0 and is_t1:
                xm, ym = x0 + (1 - core_frac) * dx, y0 + (1 - core_frac) * dy
                segs += [(xm, ym, "slide", None), (x1, y1, "core", triple_id(x1, y1))]
            else:  # both endpoints are triple points
                xm1, ym1 = x0 + core_frac * dx, y0 + core_frac * dy
                xm2, ym2 = x0 + (1 - core_frac) * dx, y0 + (1 - core_frac) * dy
                segs += [(xm1, ym1, "core", triple_id(x0, y0)),
                         (xm2, ym2, "slide", None),
                         (x1, y1, "core", triple_id(x1, y1))]

            # Draw the labeled edge segments. Core stubs carry a per-junction
            # suffix (core_i_j_lr_jJID) so each maps to a single triple junction;
            # the right-side stub of each (edge, junction) is registered for the
            # junction-coupling Lagrange constraint (the multiplier lives on the
            # 'right' trace). Consumers select cores with a 'core_i_j_lr.*' regex.
            for (xe, ye, kind, jid) in segs:
                if kind == "core" and jid is not None:
                    name = f"core_{ii}_{jj}_{lr}_j{jid}"
                    if lr == "right" and junction_incidence is not None:
                        junction_incidence.setdefault(jid, []).append(((ii, jj), name))
                else:
                    name = f"{kind}_{ii}_{jj}_{lr}"
                wp.LineTo(xe, ye, name)

        # ------------------------------------------------------------------
        # CASE 2: External boundary edge (outer box)
        # ------------------------------------------------------------------
        else:
            edge_key = (r_idx, reg[k], reg[(k + 1) % n])
            meta = external_labels.get(edge_key) if external_labels is not None else None

            base_label = None
            prefix = None
            if isinstance(meta, dict):
                base_label = meta.get("base_label")
                prefix = meta.get("prefix")
            elif isinstance(meta, str):
                base_label = meta

            if base_label is None:
                if approx(x0, -xmax, tol) and approx(x1, -xmax, tol):
                    base_label = "left"
                elif approx(x0, xmax, tol) and approx(x1, xmax, tol):
                    base_label = "right"
                elif approx(y0, -ymax, tol) and approx(y1, -ymax, tol):
                    base_label = "bottom"
                elif approx(y0, ymax, tol) and approx(y1, ymax, tol):
                    base_label = "top"

            dx, dy = x1 - x0, y1 - y0
            start_vid = reg[k]
            end_vid = reg[(k + 1) % n]
            c0 = corner_label((x0, y0), start_vid)
            c1 = corner_label((x1, y1), end_vid)

            if prefix is None:
                if not base_label or corner_core_frac is None or (corner_by_vertex is None and corner_by_coord is None):
                    wp.LineTo(x1, y1, base_label) if base_label else wp.LineTo(x1, y1)
                    continue

                if c0 is None and c1 is None:
                    wp.LineTo(x1, y1, base_label)
                    continue

                if c0 is not None and c1 is None:
                    xm, ym = (x0 + corner_core_frac * dx,
                              y0 + corner_core_frac * dy)
                    wp.LineTo(xm, ym, c0)
                    wp.LineTo(x1, y1, base_label)
                    continue

                if c0 is None and c1 is not None:
                    xm, ym = (x0 + (1 - corner_core_frac) * dx,
                              y0 + (1 - corner_core_frac) * dy)
                    wp.LineTo(xm, ym, base_label)
                    wp.LineTo(x1, y1, c1)
                    continue

                xm1, ym1 = (x0 + corner_core_frac * dx,
                            y0 + corner_core_frac * dy)
                xm2, ym2 = (x0 + (1 - corner_core_frac) * dx,
                            y0 + (1 - corner_core_frac) * dy)
                wp.LineTo(xm1, ym1, c0)
                wp.LineTo(xm2, ym2, base_label)
                wp.LineTo(x1, y1, c1)
                continue

            # With metadata present, treat the edge like a grain boundary and
            # split into core/slide segments while preserving corner tags.
            eps = 1e-12
            cf = float(core_frac or 0.0)
            corner_frac = float(corner_core_frac or 0.0)

            is_outer_start = bool(outer_kink_vertices) and (start_vid in outer_kink_vertices) and not c0
            is_outer_end = bool(outer_kink_vertices) and (end_vid in outer_kink_vertices) and not c1

            segments = []  # list of (s0, s1, name)
            s = 0.0
            e = 1.0

            if c0 and corner_frac > eps:
                seg_end = min(s + corner_frac, e)
                if seg_end - s > eps:
                    segments.append((s, seg_end, c0))
                    s = seg_end

            if c1 and corner_frac > eps:
                seg_start = max(e - corner_frac, s)
                if e - seg_start > eps:
                    segments.append((seg_start, e, c1))
                    e = seg_start

            if is_outer_start and cf > eps:
                seg_end = min(s + cf, e)
                if seg_end - s > eps:
                    segments.append((s, seg_end, outer_core_name(prefix, (x0, y0), (x1, y1), start_vid)))
                    s = seg_end

            if is_outer_end and cf > eps:
                seg_start = max(e - cf, s)
                if e - seg_start > eps:
                    segments.append((seg_start, e, outer_core_name(prefix, (x1, y1), (x0, y0), end_vid)))
                    e = seg_start

            if e - s > eps:
                segments.append((s, e, f"slide_{prefix}"))

            segments.sort(key=lambda item: item[0])
            current_param = 0.0
            for seg_start, seg_end, name in segments:
                if seg_end <= current_param + eps:
                    continue
                target = seg_end
                x_end = x0 + target * dx
                y_end = y0 + target * dy
                wp.LineTo(x_end, y_end, name)
                current_param = target

            if current_param < 1.0 - 1e-8:
                wp.LineTo(x1, y1, f"slide_{prefix}")

    # Create the face and assign region name
    face = wp.Face()
    face.name = f"region_{r_idx}"
    return face


# -----------------------------------------------------------------------------
# Build full geometry and mesh
# -----------------------------------------------------------------------------
def build_geometry_with_region_labels(comm, points, regions, xmax, ymax,
                                      tol=1e-8, maxh=0.1,
                                      triple_pts=None, core_frac=None,
                                      corner_core_frac=None, refine_h=None,
                                      refine_frac=None, refine_cut=0.02):
    """
    Build the full multi-region geometry and generate the mesh.

    Returns
    -------
    shape : OCC Compound
        Combined geometry of all regions.
    geo : OCCGeometry
        NGSolve geometry object.
    mesh : ngsolve.Mesh
        Generated NGSolve mesh.
    faces : list
        OCC faces for each region.
    contact_pairs : dict
        Mapping {(i,j)} -> ('i_j_left', 'i_j_right') for shared region pairs.
    outer_contact_pairs : dict
        Metadata describing paired outer-boundary segments treated as grain
        boundaries. Keys identify the pair, values provide the displacement
        vector and the boundary-name prefixes for the two matching faces.
    corner_label : str or None
        Name of the first constructed outer core segment, used for optional
        corner-focused penalties.
    outer_core_labels : tuple[str]
        Ordered collection of all constructed outer core boundary names
        (duplicates removed in construction order).

    Parameters
    ----------
    maxh : float
        Global target mesh size for Netgen.
    core_frac : float or None
        Fraction of internal shared edge length to mark as 'core'.
        If None, defaults to maxh (approximate).
    corner_core_frac : float or None
        Deprecated. Kept for compatibility but ignored; outer corners are not
        labeled explicitly.
    """
    if core_frac is None:
        core_frac = maxh  # default small fraction ~ mesh size

    # Corner segment fraction is no longer used; force zero to avoid tagging.
    if corner_core_frac is None:
        corner_core_frac = 0.0

    pts_arr = np.array(points, dtype=float)

    shared_map, boundary_map = build_shared_edge_map(regions)
    try:
        edge_labels, corner_vertices, outer_edge_meta, outer_pairs, outer_kink_vertices = classify_external_edges(
            points, boundary_map, tol=tol)
    except ValueError:
        edge_labels = {}
        corner_vertices = {}
        outer_edge_meta = {}
        outer_pairs = []
        outer_kink_vertices = set()

    corner_map = {}

    def corner_core_name(prefix, vertex_id, other_vertex_id):
        if not prefix or vertex_id is None or other_vertex_id is None:
            return None
        tol_cmp = 1e-12
        px, py = pts_arr[vertex_id - 1]
        ox, oy = pts_arr[other_vertex_id - 1]
        dx = px - ox
        dy = py - oy
        if dx > tol_cmp:
            suffix = "upper"
        elif dx < -tol_cmp:
            suffix = "lower"
        elif dy > tol_cmp:
            suffix = "upper"
        else:
            suffix = "lower"
        return f"core_{prefix}_{suffix}"

    outer_core_label_map = defaultdict(set)
    outer_core_sequence = []
    outer_core_vertices = defaultdict(set)

    # Records, per triple junction, the incident interior edges and the name of
    # their right-side core stub at that junction. Consumed by the GB pressure
    # junction-coupling (exact-Lagrange) assembly in main.py.
    junction_incidence = {}

    # Construct each region’s face, labeling its edges
    faces = [
        face_from_region(points, reg, r_idx, shared_map,
                         xmax, ymax, tol,
                         triple_pts=triple_pts,
                         core_frac=core_frac,
                                                 corner_core_frac=corner_core_frac,
                         corner_vertices=corner_map,
                         external_labels=outer_edge_meta,
                         outer_kink_vertices=outer_kink_vertices,
                         outer_core_label_map=outer_core_label_map,
                         outer_core_sequence=outer_core_sequence,
                         outer_core_vertices=outer_core_vertices,
                         junction_incidence=junction_incidence)
        for r_idx, reg in enumerate(regions, start=1)
    ]
    def select_label_for_vertex(vertex_id):
        if vertex_id is None:
            return None
        candidate_labels = outer_core_vertices.get(vertex_id)
        if not candidate_labels:
            return None
        for name in outer_core_sequence:
            if name in candidate_labels:
                return name
        return next(iter(candidate_labels))

    corner_labels = []
    if outer_core_vertices:
        def lower_left_key(vid):
            x_val, y_val = pts_arr[vid - 1]
            return (x_val, y_val)

        def upper_right_key(vid):
            x_val, y_val = pts_arr[vid - 1]
            return (x_val, y_val)

        primary_vertex = min(outer_core_vertices.keys(), key=lower_left_key)
        primary_label = select_label_for_vertex(primary_vertex)
        if primary_label:
            corner_labels.append(primary_label)

        remaining_vertices = [vid for vid in outer_core_vertices.keys() if vid != primary_vertex]
        if remaining_vertices:
            secondary_vertex = max(remaining_vertices, key=upper_right_key)
            secondary_label = select_label_for_vertex(secondary_vertex)
            if secondary_label and secondary_label not in corner_labels:
                corner_labels.append(secondary_label)

    def paired_corner_label(label):
        if not label or not label.startswith("core_"):
            return None
        body = label[5:]
        prefix, sep, suffix = body.rpartition("_")
        if not sep:
            return None
        if prefix.endswith("_minus"):
            twin_prefix = prefix[:-6] + "_plus"
        elif prefix.endswith("_plus"):
            twin_prefix = prefix[:-5] + "_minus"
        else:
            return None
        return f"core_{twin_prefix}_{suffix}"

    expanded_corner_labels = []
    seen_corner_labels = set()
    for label in corner_labels:
        if label and label not in seen_corner_labels:
            expanded_corner_labels.append(label)
            seen_corner_labels.add(label)
        twin = paired_corner_label(label)
        if twin and twin not in seen_corner_labels:
            expanded_corner_labels.append(twin)
            seen_corner_labels.add(twin)

    corner_label = tuple(expanded_corner_labels)
    corner_label_set = set(expanded_corner_labels)

    # Build region contact pair map for left/right boundary names
    contact_pairs = {}
    for ukey, owners in shared_map.items():
        i, j = owners[0][0], owners[1][0]
        ii, jj = (i, j) if i < j else (j, i)
        contact_pairs[(ii, jj)] = (f"{ii}_{jj}_left", f"{ii}_{jj}_right")

    # Combine all faces without fusing → preserves left/right boundaries
    shape = Compound(faces)

    # ------------------------------------------------------------------
    # Local mesh refinement at corners AND interior triple junctions
    # ------------------------------------------------------------------
    # Target element size at the refinement features. `refine_h` overrides the
    # core_frac-derived default; this decouples mesh refinement from `core_frac`,
    # which independently sets the junction-coupling stub length. Diffusion
    # concentrates over a length l ~ ω^{-1/3} at the triple junctions
    # (Rudge 2025, eq 3.4 / fig 8), so the high-ω attenuation needs fine elements
    # there — not globally.
    #
    # Refinement is applied by shrinking the target size on the short 'core' edge
    # stubs (named core_*). These are exactly the segments abutting (i) the
    # interior triple junctions (core_i_j_lr_jJID) and (ii) the outer corners
    # (core_<prefix>_<suffix>). We set it on the OCC *edges*: edge .maxh refines
    # along interior edges (vertex .maxh only bites on boundary vertices, and
    # OCCGeometry has no SetLocalH — the old corner-label call silently failed).
    smallh = refine_h if refine_h is not None else min(maxh, core_frac * 0.5)

    # Per-facet PROPORTIONAL refinement of short interior grain-boundary facets
    # (experimental; only active when refine_frac is given). The default loop
    # below sizes core stubs by an ABSOLUTE `smallh`, so a sub-maxh sliver is
    # never resolved (its `slide` middle keeps the global maxh) and its junction
    # stub (length core_frac*L) is far smaller than the element on it. Here we
    # instead set the element size PROPORTIONAL to each facet's own length,
    # `L*refine_frac`, on BOTH its `core_` and `slide_` edges, for facets shorter
    # than `refine_cut`. That resolves the sliver AND brings stub~element so the
    # (ideal, point-like) junction constraint is well-posed there. Long facets and
    # the outer box are untouched. Grain ids in the edge names (core/slide_ii_jj_*)
    # are 1-based region indices -> build an L map keyed by the same.
    fac_len = {}
    if refine_frac is not None:
        import re as _re
        from collections import defaultdict as _dd
        _own = _dd(list); _crd = {}
        for _ri, _reg in enumerate(regions, start=1):
            _n = len(_reg)
            for _k in range(_n):
                _key = tuple(sorted((_reg[_k], _reg[(_k + 1) % _n])))
                _own[_key].append(_ri)
                _crd[_key] = (points[_key[0] - 1], points[_key[1] - 1])
        for _key, _regs in _own.items():
            if len(_regs) == 2:
                (_p, _q) = _crd[_key]
                fac_len[tuple(sorted(_regs))] = float(
                    ((_p[0] - _q[0]) ** 2 + (_p[1] - _q[1]) ** 2) ** 0.5)
        _gbpat = _re.compile(r"^(?:core|slide)_(\d+)_(\d+)_")

    for e in shape.edges:
        nm = e.name
        if not nm:
            continue
        if refine_frac is not None and (nm.startswith("core_") or nm.startswith("slide_")):
            _m = _gbpat.match(nm)
            if _m:
                _gp = tuple(sorted((int(_m.group(1)), int(_m.group(2)))))
                _L = fac_len.get(_gp)
                if _L is not None and _L < refine_cut:
                    e.maxh = max(_L * refine_frac, 1e-7)   # proportional; floor
                    continue
        if nm.startswith("core_"):
            e.maxh = smallh

    geo   = OCCGeometry(shape, dim=2)
    mesh  = Mesh(geo.GenerateMesh(maxh=maxh, comm=comm))

    # Build outer boundary contact pairs with orientation metadata
    outer_contact_pairs = {}
    for pair in outer_pairs:
        pair_id = pair.get("pair_id")
        displacement = pair.get("displacement")
        edges = pair.get("edges", [])
        if pair_id is None or displacement is None or len(edges) != 2:
            continue

        side_map = {edge.get("side"): edge for edge in edges}
        minus_edge = side_map.get("minus")
        plus_edge = side_map.get("plus")
        if not minus_edge or not plus_edge:
            continue

        minus_prefix = minus_edge.get("prefix")
        plus_prefix = plus_edge.get("prefix")
        # Sort core_names consistently to ensure geometric matching between minus/plus sides
        minus_core_names = sorted([name for name in outer_core_label_map.get(minus_prefix, []) if name not in corner_label_set])
        plus_core_names = sorted([name for name in outer_core_label_map.get(plus_prefix, []) if name not in corner_label_set])

        outer_contact_pairs[f"outer_pair_{pair_id}"] = {
            "displacement": tuple(displacement),
            "minus": {
                "label": "minus",
                "prefix": minus_prefix,
                "core_names": minus_core_names,
            },
            "plus": {
                "label": "plus",
                "prefix": plus_prefix,
                "core_names": plus_core_names,
            },
        }

    return (shape, geo, mesh, faces, contact_pairs,
        outer_contact_pairs,
        corner_label,
        tuple(outer_core_sequence),
        junction_incidence)


# -----------------------------------------------------------------------------
# Identify triple junction vertices
# -----------------------------------------------------------------------------
def find_triple_vertices(points, regions):
    """
    Find points that belong to ≥3 distinct regions (triple junctions).
    Returns list of (x, y) coordinates.
    """
    vertex_regions = {i: set() for i in range(1, len(points) + 1)}
    for r_idx, reg in enumerate(regions, start=1):
        for v in reg:
            vertex_regions[v].add(r_idx)
    triple_pts = [points[i - 1] for i, regs in vertex_regions.items()
                  if len(regs) >= 3]
    return triple_pts



#-----------------------------------------------------------------------------
#Definition of a periodic RVE mesh with 100 grains
#-----------------------------------------------------------------------------
def MakeMesh(pts,regions,maxh, comm, core_frac=None, corner_core_frac=None,
             refine_h=None, refine_frac=None, refine_cut=0.02):
    """Construct an RVE mesh with full boundary metadata for the supplied tessellation.

    ``refine_h`` sets the target element size at the outer corners and interior
    triple junctions, independently of ``core_frac`` (which sets the
    junction-coupling stub length). If None, defaults to ``min(maxh,
    core_frac*0.5)`` (back-compatible).

    ``refine_frac`` (experimental): when given, interior grain-boundary facets
    shorter than ``refine_cut`` get element size ``L*refine_frac`` along their
    core+slide edges (proportional, per-facet) so sub-maxh slivers are actually
    resolved. None = unchanged behaviour.
    """

    # Locate triple junctions and build geometry
    triple_pts = find_triple_vertices(pts, regions)
    (shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label,
     outer_core_labels, junction_incidence) = build_geometry_with_region_labels(
        comm=comm,
        points=pts,
        regions=regions,
        xmax=1,
        ymax=1,
        tol=1e-8,
        maxh=maxh,
        triple_pts=triple_pts,
        core_frac=core_frac,
        corner_core_frac=corner_core_frac,
        refine_h=refine_h,
        refine_frac=refine_frac,
        refine_cut=refine_cut,
    )
    return (shape, geo, mesh, faces, contact_pairs,
        outer_contact_pairs,
        corner_label,
        outer_core_labels,
        junction_incidence)

'''# -----------------------------------------------------------------------------
# Definition of the hexagonal RVE mesh
# -----------------------------------------------------------------------------
def HexMesh(maxh, comm, core_frac=None, corner_core_frac=None):
    """
    Generate the full hexagonal RVE mesh with labeled boundaries.
    Returns (shape, geo, mesh, faces, contact_pairs, outer_contact_pairs,
    corner_label, outer_core_labels).

    Parameters
    ----------
    maxh : float
        Global target mesh size.
    core_frac : float or None
        Length fraction for internal core segments along grain boundaries.
    corner_core_frac : float or None
        Length fraction for corner segments LB, LT, RB. If None, it will
        default to core_frac inside build_geometry_with_region_labels.
    """
    a = np.sqrt(3)
    pts = [
        (0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
        (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
        (3, 0), (3, a/4), (3, a), (9/4, a),
        (3/4, a), (0, a)
    ]
    # Shift to center domain around origin
    pts1 = [(x/9.65, y/9.65) for (x, y) in pts]

    # Define polygonal regions (each grain) by vertex indices
    regions = [
        (1, 2, 3, 4),
        (2, 5, 6, 7, 8, 3),
        (5, 9, 10, 6),
        (6, 10, 11, 12, 7),
        (8, 7, 12, 13),
        (4, 3, 8, 13, 14)
    ]

    # Locate triple junctions and build geometry
    triple_pts = find_triple_vertices(pts1, regions)
    shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label, outer_core_labels = build_geometry_with_region_labels(
        comm=comm,
        points=pts1,
        regions=regions,
        xmax=1.5,
        ymax=a * 0.5,
        tol=1e-8,
        maxh=maxh,
        triple_pts=triple_pts,
        core_frac=core_frac,
        corner_core_frac=corner_core_frac
    )
    return (shape, geo, mesh, faces, contact_pairs,
        outer_contact_pairs,
        corner_label,
        outer_core_labels)'''