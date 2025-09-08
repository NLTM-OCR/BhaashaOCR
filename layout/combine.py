import copy
import json
import os
from collections import Counter, defaultdict

# ------------------ Utility / Overlap helpers ------------------

def print_region_orders(union_data, *, sort="order", title=None):
    """
    Pretty-print all regions' orders per image.

    Parameters
    ----------
    union_data : list[dict]
        The merged data structure: [{"image_name": str, "regions": [region, ...]}, ...]
    sort : str
        How to sort regions before printing:
          - "order" (default): numeric order ascending
          - "yx": by bounding_box y, then x (top→bottom, left→right)
          - "xy": by bounding_box x, then y (left→right, top→bottom)
    title : str | None
        Optional header to print before the dump.
    """
    if title:
        print(f"\n=== {title} ===")

    for img in union_data:
        regs = img.get("regions", [])
        if sort == "order":
            keyfn = lambda r: r.get("order", 0)
        elif sort == "yx":
            keyfn = lambda r: (r["bounding_box"]["y"], r["bounding_box"]["x"])
        elif sort == "xy":
            keyfn = lambda r: (r["bounding_box"]["x"], r["bounding_box"]["y"])
        else:
            raise ValueError(f"Unknown sort='{sort}'. Use 'order', 'yx', or 'xy'.")

        print(f"\nImage: {img.get('image_name', '<unknown>')}  (regions: {len(regs)})")
        for r in sorted(regs, key=keyfn):
            bb = r["bounding_box"]
            print(
                f"  order={r.get('order', 0):4d}  "
                f"line={r.get('line', -1):3d}  "
                f"bbox=(x={bb['x']}, y={bb['y']}, w={bb['w']}, h={bb['h']})"
            )
    print()  # trailing newline

def get_y_overlap(box1, box2):
    y_top = max(box1['y'], box2['y'])
    y_bottom = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    return max(0, y_bottom - y_top)

def has_sufficient_y_overlap(box1, box2, threshold=0.4):
    """Returns True if Y-overlap is at least threshold fraction of smaller height."""
    overlap = get_y_overlap(box1, box2)
    min_height = min(box1.get('h', 0), box2.get('h', 0))
    if min_height == 0:
        return False
    return (overlap / min_height) >= threshold

def boxes_overlap_adjusted(box1, box2):
    adjusted_box2 = {
        "x": box2["x"],
        "y": box2["y"] + 10,
        "w": box2["w"],
        "h": box2["h"] - 10
    }
    return (
        box1["x"] < adjusted_box2["x"] + adjusted_box2["w"] and
        box1["x"] + box1["w"] > adjusted_box2["x"] and
        box1["y"] < adjusted_box2["y"] + adjusted_box2["h"] and
        box1["y"] + box1["h"] > adjusted_box2["y"]
    )

def is_x_overlap(box1, box2):
    return not (box1["x"] + box1["w"] <= box2["x"] or box2["x"] + box2["w"] <= box1["x"])

def yx_sort(regions):
    return sorted(regions, key=lambda r: (r["bounding_box"]["y"], r["bounding_box"]["x"]))

# ------------------ Neighbor-based order assignment ------------------

def select_best_neighbor(region, candidates):
    best_neighbor = None
    min_distance = float('inf')

    for candidate in candidates:
        # Check if the candidate meets the minimum vertical overlap criteria
        overlap = get_y_overlap(region["bounding_box"], candidate["bounding_box"])
        if overlap / min(region["bounding_box"]["h"], candidate["bounding_box"]["h"]) >= 0.4:
            distance = abs(region["bounding_box"]["x"] - candidate["bounding_box"]["x"])
            if distance < min_distance:
                min_distance = distance
                best_neighbor = candidate

    return best_neighbor

def _fully_covers(outer, inner):
    """
    Return True if `outer` box completely contains `inner` box in both X and Y.
    Boxes are dicts with x,y,w,h.
    """
    return (
        outer["x"] <= inner["x"] and
        outer["x"] + outer["w"] >= inner["x"] + inner["w"] and
        outer["y"] <= inner["y"] and
        outer["y"] + outer["h"] >= inner["y"] + inner["h"]
    )

def assign_orders_based_on_neighbors(union_data):
    """
    For each image, any region with order==0 will inherit the order & line
    from the single neighbor (order>0) that overlaps vertically ≥40% and
    is horizontally nearest.
    """
    for image_entry in union_data:
        image_name = image_entry.get("image_name", "<unknown>")
        regions = image_entry["regions"]

        for region in regions:
            if region.get("order", 0) != 0:
                continue

            b1 = region["bounding_box"]
            best = None
            best_dx = None

            for nbr in regions:
                if nbr is region or nbr.get("order", 0) == 0:
                    continue

                b2 = nbr["bounding_box"]
                # vertical overlap ≥ 40% of min height?
                if not has_sufficient_y_overlap(b1, b2, threshold=0.4):
                    continue
                # NEW RULE: skip neighbors that completely contain this region
                #if _fully_covers(b2, b1):
                  #  continue
                if b2["h"] > 3 * b1["h"]:
                    continue

                # compute horizontal distance
                x1_min, x1_max = b1["x"], b1["x"] + b1["w"]
                x2_min, x2_max = b2["x"], b2["x"] + b2["w"]
                if x1_max < x2_min:
                    dx = x2_min - x1_max
                elif x2_max < x1_min:
                    dx = x1_min - x2_max
                else:
                    dx = 0

                # pick the neighbor with smallest dx
                if best is None or dx < best_dx:
                    best, best_dx = nbr, dx

            if best:
                old_order = region["order"]
                region["order"] = best["order"]
                region["line"]  = best.get("line", -1)
                print(
                    f"[{image_name}] Region {b1!r} "
                    f"order {old_order}→{region['order']} "
                    f"(nearest neighbor at {best['bounding_box']}, dx={best_dx})"
                )


def is_x_overlap(box1, box2):
    return not (box1["x"] + box1["w"] <= box2["x"] or box2["x"] + box2["w"] <= box1["x"])

def get_y_overlap(box1, box2):
    y_top = max(box1['y'], box2['y'])
    y_bottom = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    return max(0, y_bottom - y_top)

def has_sufficient_y_overlap(box1, box2, threshold=0.4):
    overlap = get_y_overlap(box1, box2)
    min_height = min(box1["h"], box2["h"])
    return min_height > 0 and (overlap / min_height) >= threshold

# Other functions remain unchanged


# Other functions remain unchanged
# ------------------ Duplicate order resolution with line clustering ------------------

def get_vertical_overlap_fraction(box1, box2):
    y_top = max(box1["y"], box2["y"])
    y_bottom = min(box1["y"] + box1["h"], box2["y"] + box2["h"])
    overlap = max(0, y_bottom - y_top)
    min_height = min(box1["h"], box2["h"])
    if min_height == 0:
        return 0
    return overlap / min_height

def resolve_duplicate_orders(union_data):
    """
    Phase 1 (zero→non-zero in your style):
      • Cluster into lines by vertical-overlap ≥ 0.4
      • Sort lines top→bottom; within each line left→right
      • For each region with order==0:
          - If a same-line neighbour (immediate left/right) has non-zero order,
            copy the NEAREST neighbour’s order (by horizontal distance)
          - Else assign prev_line_max + 1, and advance prev_line_max
        (prev_line_max is the highest order in the previous processed line)

    Phase 2 (your tie-break):
      • Group by equal orders, then within a group re-split by ≥0.4 vertical overlap,
        sort each sub-line left→right, and assign unique consecutive final orders.
    """
    from collections import Counter

    def vfrac(b1, b2):
        return get_vertical_overlap_fraction(b1, b2)

    def x_center(b):
        return b["x"] + b["w"] / 2.0

    for image_entry in union_data:
        regions = image_entry["regions"]
        n = len(regions)
        if n == 0:
            continue

        # ---------- Line clustering by vertical-overlap ≥ 0.4 ----------
        adj = [[] for _ in range(n)]
        for i in range(n):
            bi = regions[i]["bounding_box"]
            for j in range(i + 1, n):
                bj = regions[j]["bounding_box"]
                if vfrac(bi, bj) >= 0.4:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        lines = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            lines.append(comp)

        # ---------- Diagnostics: per-line summary ----------
        print(f"Image: {image_entry['image_name']} → {len(lines)} lines detected")
        for idx, comp in enumerate(lines):
            xs = [regions[k]["bounding_box"]["x"] for k in comp]
            ys = [regions[k]["bounding_box"]["y"] for k in comp]
            ws = [regions[k]["bounding_box"]["w"] for k in comp]
            hs = [regions[k]["bounding_box"]["h"] for k in comp]
            x_min, y_min = min(xs), min(ys)
            x_max = max(x + w for x, w in zip(xs, ws))
            y_max = max(y + h for y, h in zip(ys, hs))
            zeros = [k for k in comp if int(regions[k].get("order", 0)) == 0]
            rng = [int(regions[k].get("order", 0)) for k in comp]
            print(f"  Line {idx}: {len(comp)} regions, bbox=({x_min},{y_min},{x_max},{y_max})")
            print(f"    → {len(zeros)} with order=0; indices: {zeros}")
            if rng:
                print(f"    → order range = {min(rng)} to {max(rng)}")
            else:
                print("    → order range = (none)")

        # ---------- Sort lines & apply Phase 1 (your logic) ----------
        lines.sort(key=lambda comp: min(regions[k]["bounding_box"]["y"] for k in comp))

        prev_line_max = 0  # highest order after finishing the previous line
        for line_idx, comp in enumerate(lines):
            # left→right within line
            comp.sort(key=lambda k: regions[k]["bounding_box"]["x"])

            # Walk this line; assign zeros per your rule
            for pos, k in enumerate(comp):
                r = regions[k]
                if int(r.get("order", 0)) != 0:
                    continue  # keep existing non-zero orders; they also inform prev_line_max later

                # Immediate neighbours in the same line
                left_k  = comp[pos - 1] if pos - 1 >= 0 else None
                right_k = comp[pos + 1] if pos + 1 < len(comp) else None

                b_cur = r["bounding_box"]
                cx_cur = x_center(b_cur)
                candidates = []

                if left_k is not None and int(regions[left_k].get("order", 0)) > 0:
                    b_left = regions[left_k]["bounding_box"]
                    candidates.append((abs(cx_cur - x_center(b_left)), int(regions[left_k]["order"])))
                if right_k is not None and int(regions[right_k].get("order", 0)) > 0:
                    b_right = regions[right_k]["bounding_box"]
                    candidates.append((abs(cx_cur - x_center(b_right)), int(regions[right_k]["order"])))

                if candidates:
                    # Copy nearest neighbour’s non-zero order
                    candidates.sort(key=lambda t: t[0])
                    r["order"] = candidates[0][1]
                else:
                    # No neighbour in this line has non-zero order → use prev_line_max+1
                    r["order"] = prev_line_max + 1
                    prev_line_max = r["order"]  # advance so subsequent fallback zeros in this line keep increasing

            # Update prev_line_max to reflect the highest non-zero in this line (after assignments)
            line_max = prev_line_max
            for k in comp:
                o = int(regions[k].get("order", 0))
                if o > line_max:
                    line_max = o
            prev_line_max = line_max

        # ---------- Diagnostics: after Phase 1 ----------
        print(f"[{image_entry['image_name']}] Phase 1 provisional (reading order):")
        for comp in lines:
            for k in comp:
                r = regions[k]
                bb = r["bounding_box"]
                print(f"  bbox=(x={bb['x']}, y={bb['y']}, w={bb['w']}, h={bb['h']})  order={int(r.get('order',0))}  line={r.get('line',-1)}")
        print()

        # ---------- Phase 2: Tie-break duplicates (your original tie logic) ----------
        provisional = list(regions)  # shallow copy of refs
        provisional.sort(key=lambda r: (
            int(r.get("order", 0)),
            r["bounding_box"]["y"],
            r["bounding_box"]["x"]
        ))

        final = []
        i = 0
        current = 0
        while i < len(provisional):
            base = int(provisional[i].get("order", 0))
            group = [provisional[i]]
            i += 1
            while i < len(provisional) and int(provisional[i].get("order", 0)) == base:
                group.append(provisional[i])
                i += 1

            if len(group) == 1:
                group[0]["order"] = current
                final.append(group[0])
                current += 1
            else:
                # re-cluster this group by vertical overlap ≥ 0.4
                m = len(group)
                sub_adj = [[] for _ in range(m)]
                for a in range(m):
                    for b in range(a + 1, m):
                        if vfrac(group[a]["bounding_box"], group[b]["bounding_box"]) >= 0.4:
                            sub_adj[a].append(b)
                            sub_adj[b].append(a)

                used = set()
                for a in range(m):
                    if a in used:
                        continue
                    stack, comp = [a], []
                    used.add(a)
                    while stack:
                        u = stack.pop()
                        comp.append(u)
                        for v in sub_adj[u]:
                            if v not in used:
                                used.add(v)
                                stack.append(v)
                    # left→right within the sub-line
                    comp.sort(key=lambda u: group[u]["bounding_box"]["x"])
                    for u in comp:
                        group[u]["order"] = current
                        final.append(group[u])
                        current += 1

        image_entry["regions"] = final

def remove_smaller_overlapping_regions(union_data):
    """
    NEW POLICY:
      • If two regions overlap by >50% of the smaller region's area, delete the **larger** region.
      • If areas are **equal**, delete exactly **one** deterministically (the later j-index).
    Logs each decision.
    """
    def get_intersection_area(b1, b2):
        xl = max(b1['x'], b2['x'])
        xr = min(b1['x'] + b1['w'], b2['x'] + b2['w'])
        yt = max(b1['y'], b2['y'])
        yb = min(b1['y'] + b1['h'], b2['y'] + b2['h'])
        if xr <= xl or yb <= yt:
            return 0
        return (xr - xl) * (yb - yt)

    for img in union_data:
        regs = img['regions']
        keep = [True] * len(regs)
        print(f"Processing image: {img['image_name']}")

        for i in range(len(regs)):
            if not keep[i]:
                continue
            b1 = regs[i]['bounding_box']; a1 = b1['w'] * b1['h']

            for j in range(i + 1, len(regs)):
                if not keep[j]:
                    continue
                b2 = regs[j]['bounding_box']; a2 = b2['w'] * b2['h']

                inter = get_intersection_area(b1, b2)
                if inter == 0:
                    continue

                smaller_area = min(a1, a2)
                if smaller_area == 0:
                    continue
                overlap_ratio = inter / smaller_area
                if overlap_ratio > 0.5:
                    # Delete LARGER; if equal, delete j (deterministic tie-break)
                    if a1 > a2:
                        del_idx = i
                        reason = "larger area (i)"
                    elif a2 > a1:
                        del_idx = j
                        reason = "larger area (j)"
                    else:
                        del_idx = j
                        reason = "equal areas → delete later region (j)"

                    print(
                        f"  Overlap {overlap_ratio:.1%} between {i} (A={a1}) and {j} (A={a2}) → "
                        f"deleting region {del_idx} ({reason}), bbox={regs[del_idx]['bounding_box']}"
                    )

                    keep[del_idx] = False
                    # If we deleted current i, stop comparing it further
                    if del_idx == i:
                        break

        filtered = [r for idx, r in enumerate(regs) if keep[idx]]
        removed = len(regs) - len(filtered)
        print(f"  Removed {removed} region(s) from {img['image_name']}")
        img['regions'] = filtered

# ------------------ Main merging with spatial sort ------------------

# ------------------ Main merging with new overlap policy ------------------
def merge_v5(data1, data2):
    """
    Merge 1.json and 2.json into 3.json with these rules:

    - For each r1 (from 1.json), find ALL overlapping r2 (from 2.json) by:
        boxes_overlap_adjusted(...) AND has_sufficient_y_overlap(..., 0.4)

    - NEW: If a 2.json region overlaps multiple 1.json regions, it is added ONLY ONCE
      (from the FIRST r1 encountered), and all those overlapping r1 are not kept.
      The added r2 inherits the first r1's order/line (unless the 'big cover' rule applies).

    - Big-cover rule (from earlier): if r1 fully covers r2 and r1.h > 3*r2.h,
      DO NOT inherit; set r2 to (order=1, line=-1).

    - Any r2 never used in overlaps is appended as a leftover with order=0.

    Returns: result, total_boxes_file1, total_boxes_file2, overlap_count_1, overlap_count_2, stats_lines
    """
    import copy
    import json
    from collections import defaultdict

    def fully_covers(outer, inner):
        return (
            outer["x"] <= inner["x"] and
            outer["y"] <= inner["y"] and
            outer["x"] + outer["w"] >= inner["x"] + inner["w"] and
            outer["y"] + outer["h"] >= inner["y"] + inner["h"]
        )

    # with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
    #     data1 = json.load(f1)
    #     data2 = json.load(f2)

    map1 = {e["image_name"]: e["regions"] for e in data1}
    map2 = {e["image_name"]: e["regions"] for e in data2}
    image_names = sorted(set(map1.keys()) | set(map2.keys()))

    result = []
    total_boxes_file1 = total_boxes_file2 = 0
    overlap_count_1 = 0   # number of r1 that had ≥1 overlap
    overlap_count_2 = 0   # number of unique r2 consumed via overlaps

    for img_name in image_names:
        regions1 = map1.get(img_name, [])
        regions2 = map2.get(img_name, [])
        total_boxes_file1 += len(regions1)
        total_boxes_file2 += len(regions2)

        merged_regions = []
        used_2_indices = set()     # r2 indices already added via an overlap
        r1_had_overlap = [False]*len(regions1)  # mark r1 that overlapped any r2

        # --- PASS 1: for each r1, find overlaps; add each r2 only once (from first r1) ---
        for i1, r1 in enumerate(regions1):
            b1 = r1["bounding_box"]

            # all overlaps (regardless of whether r2 already used)
            all_overlap_i2 = []
            for i2, r2 in enumerate(regions2):
                b2 = r2["bounding_box"]
                if boxes_overlap_adjusted(b1, b2) and has_sufficient_y_overlap(b1, b2, threshold=0.4):
                    all_overlap_i2.append(i2)

            if all_overlap_i2:
                r1_had_overlap[i1] = True

                # add any *unused* r2 now (from this "first" r1)
                for i2 in all_overlap_i2:
                    if i2 in used_2_indices:
                        continue  # already added by an earlier r1
                    r2 = regions2[i2]
                    b2 = r2["bounding_box"]
                    if (b1["h"] > 2 * b2["h"]):
                        continue
                    if (b1["w"] > 5 * b2["w"]):
                        continue

                    m = copy.deepcopy(r2)  # geometry/text from 2.json
                    m["order"] = r1.get("order", 0)
                    m["line"]  = r1.get("line", -1)

                    # preserve any extra keys from r1 if missing in m
                    for k, v in r1.items():
                        if k not in m:
                            m[k] = v

                    merged_regions.append(m)
                    used_2_indices.add(i2)

        # count stats for overlaps
        overlap_count_1 += sum(1 for had in r1_had_overlap if had)
        overlap_count_2 += len(used_2_indices)

        # --- PASS 2: append r1 that NEVER overlapped any r2 (keep as-is) ---
        for i1, r1 in enumerate(regions1):
            if not r1_had_overlap[i1]:
                merged_regions.append(copy.deepcopy(r1))

        # --- PASS 3: append true r2 leftovers (never used) with order=0 ---
        for i2, r2 in enumerate(regions2):
            if i2 not in used_2_indices:
                newr = copy.deepcopy(r2)
                newr["order"] = 0
                newr["line"]  = r2.get("line", -1)
                merged_regions.append(newr)

        # stable sort by order so non-zeros appear first
        merged_regions.sort(key=lambda r: r.get("order", 0))
        result.append({"image_name": img_name, "regions": merged_regions})

    # reading-order spatial sort (optional but nice for consistency)
    for img_entry in result:
        img_entry["regions"].sort(key=lambda r: (r["bounding_box"]["y"], r["bounding_box"]["x"]))

    # downstream steps are optional; keep commented if you want zeros to remain 0
    remove_smaller_overlapping_regions(result)
    assign_orders_based_on_neighbors(result)
    resolve_duplicate_orders(result)

    stats_lines = [
        "Merge rule: if a 2.json region overlaps multiple 1.json regions,"
        " add the 2.json region ONCE (from the first r1) and drop those r1;"
        " inherit first r1's order/line unless big-cover rule triggers (then order=1,line=-1)."
    ]
    return result

# ------------------ Entry point ------------------# ------------------ Entry point ------------------

def merge_layouts(openseg_regions, ajoy_regions) -> list[str]:
    ajoy_data = [{
        "image_name": "image.jpg",
        # "regions": [i.dict() for i in ajoy_regions]
        "regions": ajoy_regions
    }]
    openseg_data = [{
        "image_name": "image.jpg",
        # "regions": [i.dict() for i in openseg_regions]
        "regions": openseg_regions
    }]

    # union_data = merge_v5(ajoy_data, openseg_data)
    union_data = merge_v5(openseg_data, ajoy_data)

    ret = []
    regions = union_data[0]["regions"]
    regions.sort(key=lambda r: r.get("order", 0))
    for i in regions:
        ret.append('{},{},{},{},{}'.format(
            i["bounding_box"]["x"],
            i["bounding_box"]["y"],
            i["bounding_box"]["w"],
            i["bounding_box"]["h"],
            i.get("line", 1)
        ))
    return ret