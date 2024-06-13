# Printed End-to-End Page Recognition for Indic Languages

## Pretrained Layout Models:
- You can find the pretrained model for the DocTR Layout under the [Assests]().

## Pretrained OCR Models:
- You can find the pretrained models for V4 printed for 13 languages under the [Assets]().

## Setup
- Using Python = 3.10+
- Install Dependencies `pip install -r requirements.txt`

## Inference (New)

For Inference please call the `infer.py` file. The OCR outputs are generated in JSON file and saved in the directory specified by `out_dir` argument.

### Arguments
* `--layout_pretrained`: Path to Layout pretrained model file (.pt)
* `--ocr_pretrained`: Path to OCR pretrained model file (.pth)
* `--image_path`: Path to the input image for Inference
* `--out_dir`: Path to folder where JSON OCR output is saved.
* `--language`: language of the input images

### Example

```bash
python infer.py \
  --layout_pretrained=/home/ocr/layout_model.pt \
  --ocr_pretrained=/home/ocr/model/best_cer.pth \
  --image_path=/home/ocr/image.jpg \
  --language=bengali --out_dir=/home/ocr/out
```

## Contact

You can contact **[Ajoy Mondal](mailto:ajoy.mondal@iiit.ac.in)** or **[Krishna Tulsyan](mailto:krishna.tulsyan@research.iiit.ac.in)** for any issues or feedbacks.

## Citation

```
@InProceedings{iiit_hw,
	author="Gongidi, Santhoshini and Jawahar, C. V.",
	editor="Llad{\'o}s, Josep and Lopresti, Daniel and Uchida, Seiichi",
	title="iiit-indic-hw-words: A Dataset for Indic Handwritten Text Recognition",
	booktitle="Document Analysis and Recognition -- ICDAR 2021",
	year="2021",
	pages="444--459"
}
```
