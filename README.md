# Printed End-to-End Page Recognition for Indic Languages

## Pretrained Models:
- You can find the pretrained model for End-to-End Recognition under the [Assests]().

## Setup
- Using Python = 3.10+
- Install Dependencies `pip install -r requirements.txt`

## Inference (New)

For Inference please call the `infer.py` file. The OCR outputs are generated in JSON file and saved in the directory specified by `out_dir` argument.

### Arguments
* `--pretrained`: Path to pretrained folder containing layout/ocr model files (as unzipped from the Assets downloads)
* `--image_path`: Path to the input image for Inference
* `--out_dir`: Path to folder where JSON OCR output is saved.
* `--language`: language of the input images

### Example

```bash
python infer.py \
  --pretrained=/home/ocr/bengali \
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
