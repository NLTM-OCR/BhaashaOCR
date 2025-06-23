import shutil
from tempfile import TemporaryDirectory
from os.path import basename, join

from ultralytics import YOLO

def perform_v4x(image_path, pretrained) -> list[str]:
    try:
        print(f'Processing V4x for file: {basename(image_path)}')
        model_path = join(pretrained, 'layout_v4x.pt')
        model = YOLO(model_path)
        with TemporaryDirectory() as temp_dir:
            shutil.copy(image_path, temp_dir)
            results = model.predict(source=temp_dir, max_det=4000)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist() # type: ignore
        bboxes = [tuple(i) for i in bboxes]
        regions = []
        for line, i in enumerate(bboxes):
            x1, y1, x2, y2 = i
            w = x2 - x1
            h = y2 - y1
            regions.append(','.join(list(map(str, [
                x1, y1, w, h,
                line + 1,
            ]))))
        return regions
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
        return []