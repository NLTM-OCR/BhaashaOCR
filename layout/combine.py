import json
from collections import defaultdict



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

def assign_orders_based_on_neighbors(union_data):
    for image_entry in union_data:
        regions = image_entry["regions"]
        for r1 in regions:
            if r1.get("order", -1) != 0:
                continue

            box1 = r1["bounding_box"]
            y1 = box1["y"]
            y1_min, y1_max = y1, y1 + box1["h"]
            x1 = box1["x"]

            L = [r for r in regions if r != r1 and not (
                r["bounding_box"]["y"] + r["bounding_box"]["h"] < y1_min or
                r["bounding_box"]["y"] > y1_max)]

            L1 = [r for r in L if r["bounding_box"]["x"] < x1 and r.get("order", 0) > 0]
            L2 = [r for r in L if r["bounding_box"]["x"] > x1 and r.get("order", 0) > 0]

            if L1:
                R2 = max(L1, key=lambda r: r["bounding_box"]["x"])
                r1["order"] = R2["order"]
                r1["line"] = R2.get("line", -1)
            elif L2:
                R3 = min(L2, key=lambda r: r["bounding_box"]["x"])
                r1["order"] = R3["order"]
                r1["line"] = R3.get("line", -1)
            else:
                U = [r for r in regions if r["bounding_box"]["y"] < y1 and r.get("order", 0) > 0]
                if U:
                    R4 = max(U, key=lambda r: r["bounding_box"]["y"])
                    y4_min, y4_max = R4["bounding_box"]["y"], R4["bounding_box"]["y"] + R4["bounding_box"]["h"]
                    U1 = [r for r in regions if r != R4 and not (
                        r["bounding_box"]["y"] + r["bounding_box"]["h"] < y4_min or
                        r["bounding_box"]["y"] > y4_max)]
                    U1_valid = [r for r in U1 if r.get("order", 0) > 0]
                    if U1_valid:
                        R5 = max(U1_valid, key=lambda r: r["order"])
                        r1["order"] = R5["order"]
                        r1["line"] = R5.get("line", -1) + 1
                    else:
                        r1["order"] = R4["order"]
                        r1["line"] = R4.get("line", -1) + 1
                else:
                    U3 = [r for r in regions if r["bounding_box"]["y"] > y1 and r.get("order", 0) > 0]
                    if U3:
                        R6 = min(U3, key=lambda r: r["bounding_box"]["y"])
                        y6_min, y6_max = R6["bounding_box"]["y"], R6["bounding_box"]["y"] + R6["bounding_box"]["h"]
                        U4 = [r for r in regions if r != R6 and not (
                            r["bounding_box"]["y"] + r["bounding_box"]["h"] < y6_min or
                            r["bounding_box"]["y"] > y6_max)]
                        U4_valid = [r for r in U4 if r.get("order", 0) > 0]
                        if U4_valid:
                            R7 = min(U4_valid, key=lambda r: r["order"])
                            r1["order"] = R7["order"]
                            r1["line"] = R7.get("line", -1) - 1
                        else:
                            r1["order"] = R6["order"]
                            r1["line"] = R6.get("line", -1) - 1

def resolve_duplicate_orders(union_data):
    for image_entry in union_data:
        regions = image_entry["regions"]
        regions.sort(key=lambda r: r.get("order", 0))

        new_regions = []
        i = 0
        current_order = 0

        while i < len(regions):
            # Collect all regions with same order
            base_order = regions[i]["order"]
            group = [regions[i]]
            i += 1
            while i < len(regions) and regions[i]["order"] == base_order:
                group.append(regions[i])
                i += 1

            if len(group) == 1:
                group[0]["order"] = current_order
                new_regions.append(group[0])
                current_order += 1
            else:
                # Divide group into Y-overlapping subgroups
                subgroups = []
                used = set()

                for idx, r1 in enumerate(group):
                    if idx in used:
                        continue
                    y1, h1 = r1["bounding_box"]["y"], r1["bounding_box"]["h"]
                    y2 = y1 + h1
                    subgroup = [r1]
                    used.add(idx)
                    for jdx in range(idx + 1, len(group)):
                        if jdx in used:
                            continue
                        ry1 = group[jdx]["bounding_box"]["y"]
                        ry2 = ry1 + group[jdx]["bounding_box"]["h"]
                        if not (ry2 < y1 or ry1 > y2):  # Overlaps in Y
                            subgroup.append(group[jdx])
                            used.add(jdx)
                    subgroups.append(sorted(subgroup, key=lambda r: r["bounding_box"]["x"]))

                # Sort subgroups by Y
                subgroups.sort(key=lambda g: g[0]["bounding_box"]["y"])

                for subgroup in subgroups:
                    for r in subgroup:
                        r["order"] = current_order
                        new_regions.append(r)
                        current_order += 1

        # Replace regions with deduplicated list
        image_entry["regions"] = new_regions

def remove_smaller_overlapping_regions(union_data):
    def get_intersection_area(b1, b2):
        x_left = max(b1["x"], b2["x"])
        y_top = max(b1["y"], b2["y"])
        x_right = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
        y_bottom = min(b1["y"] + b1["h"], b2["y"] + b2["h"])

        if x_right <= x_left or y_bottom <= y_top:
            return 0
        return (x_right - x_left) * (y_bottom - y_top)

    for image_entry in union_data:
        regions = image_entry["regions"]
        keep_flags = [True] * len(regions)

        for i in range(len(regions)):
            if not keep_flags[i]:
                continue
            box1 = regions[i]["bounding_box"]
            area1 = box1["w"] * box1["h"]
            for j in range(i + 1, len(regions)):
                if not keep_flags[j]:
                    continue
                box2 = regions[j]["bounding_box"]
                area2 = box2["w"] * box2["h"]

                inter_area = get_intersection_area(box1, box2)
                if inter_area == 0:
                    continue

                # Determine smaller region
                if area1 < area2:
                    smaller_idx, smaller_area = i, area1
                else:
                    smaller_idx, smaller_area = j, area2

                frac_overlap = inter_area / smaller_area
                if frac_overlap > 0.5:
                    keep_flags[smaller_idx] = False
                    if smaller_idx == i:
                        break  # No need to check further for i

        # Retain only non-overlapping largest regions
        image_entry["regions"] = [r for k, r in enumerate(regions) if keep_flags[k]]

def merge_all_regions_with_stats(data1, data2):
    map1 = {entry["image_name"]: entry["regions"] for entry in data1}
    map2 = {entry["image_name"]: entry["regions"] for entry in data2}

    result = []
    overlap_1, overlap_2 = set(), set()
    total_boxes_file1, total_boxes_file2 = 0, 0

    multiple_1_to_2 = defaultdict(list)
    multiple_2_to_1 = defaultdict(list)
    used_2_indices = defaultdict(set)

    non_overlap_1 = defaultdict(list)
    non_overlap_2 = defaultdict(list)
    skipped_large_width_2 = defaultdict(list)

    for img_name in map1:
        regions1 = map1[img_name]
        regions2 = map2.get(img_name, [])
        total_boxes_file1 += len(regions1)
        total_boxes_file2 += len(regions2)

        merged_regions = []
        invalid_indices_2 = set()
        print(type(regions1[0]), regions1[0])

        for idx1, reg1 in enumerate(regions1):
            box1 = reg1["bounding_box"]
            found = False
            for idx2, reg2 in enumerate(regions2):
                box2 = reg2["bounding_box"]
                if boxes_overlap_adjusted(box1, box2):
                    if box2["w"] > 1.3 * box1["w"]:
                        skipped_large_width_2[img_name].append((idx2, reg2.get("order", -1)))
                        invalid_indices_2.add(idx2)
                        continue

                    overlap_1.add(idx1)
                    overlap_2.add(idx2)
                    multiple_1_to_2[(img_name, idx1)].append((img_name, idx2))
                    multiple_2_to_1[(img_name, idx2)].append((img_name, idx1))
                    found = True

                    # Selection Logic
                    w1, h1 = box1["w"], box1["h"]
                    w2, h2 = box2["w"], box2["h"]

                    select_r1 = False

                    # ✅ Condition A: R1 is wider than R2
                    if (w1 >=0.95*w2 and w1 <= 3.5 * w2 and 0.4 * h2 <= h1 <= 1.5 * h2):
                        select_r1 = True

                    # ✅ Condition B: R1 is taller than R2
                    elif (h1 > h2 and h1 <= 1.4 * h2 and 0.7 * w2 <= w1 <= 1.3 * w2):
                        select_r1 = True

                    if select_r1:
                        merged = reg1.copy()
                        merged["bounding_box"] = box1
                    else:
                        merged = {
                            "order": reg1.get("order"),
                            "line": reg1.get("line"),
                            "bounding_box": box2,
                            "text": reg2.get("text", "")
                        }
                        for key in reg1:
                            if key not in merged:
                                merged[key] = reg1[key]

                    merged_regions.append(merged)
                    used_2_indices[img_name].add(idx2)

            if not found:
                merged_regions.append(reg1)
                non_overlap_1[img_name].append((idx1, reg1.get("order", -1)))

        for idx2, reg2 in enumerate(regions2):
            if idx2 not in used_2_indices[img_name] and idx2 not in invalid_indices_2:
                new_region = {
                    "order": 0,
                    "line": reg2.get("line", -1),
                    "bounding_box": reg2.get("bounding_box"),
                    "text": reg2.get("text", "")
                }
                merged_regions.append(new_region)
                non_overlap_2[img_name].append((idx2, reg2.get("order", -1)))

        merged_regions.sort(key=lambda r: r["order"])
        result.append({
            "image_name": img_name,
            "regions": merged_regions
        })

    assign_orders_based_on_neighbors(result)
    remove_smaller_overlapping_regions(result)
    resolve_duplicate_orders(result)
    stats_lines = []
    stats_lines.append("\U0001f501 Regions from 1.json with multiple overlapping matches in 2.json:")

    skipped_count = sum(len(v) for v in skipped_large_width_2.values())
    stats_lines.append(f"\n❌ Skipped regions from 2.json due to width > 1.3× and overlapping only: {skipped_count}")

    return result

def merge_layouts(ajoy_regions, openseg_regions) -> list[str]:
    ajoy_data = [{
        "image_name": "v4x.jpg",
        # "regions": [i.dict() for i in ajoy_regions]
        "regions": ajoy_regions
    }]
    openseg_data = [{
        "image_name": "openseg.jpg",
        # "regions": [i.dict() for i in openseg_regions]
        "regions": openseg_regions
    }]

    union_data = merge_all_regions_with_stats(ajoy_data, openseg_data)

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