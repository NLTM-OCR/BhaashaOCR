# Printed End-to-End Page Recognition for Indic Languages

## Pretrained Models:
- You can find the pretrained model for End-to-End Recognition under the [Assests](https://github.com/NLTM-OCR/BhaashaOCR/releases).

## Setup
- Using Python = 3.10+
- Install Tesseract (V5.4.1) from [Official source](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- Install Dependencies `pip install -r requirements.txt`

## Inference (New)

For Inference please call the `infer.py` file. The Page level OCR/Layout outputs are generated in txt file format (ocr.txt & layout.txt) and saved in the directory specified by `out_dir` argument.

### Arguments
* `--pretrained`: Path to pretrained folder containing layout/ocr model files (as unzipped from the Assets downloads)
* `--image_path`: Path to the input image for Inference
* `--out_dir`: Path to folder where OCR output is saved.

### Example

```bash
python infer.py \
  --pretrained=/home/ocr/bengali \
  --image_path=/home/ocr/image.jpg \
  --out_dir=/home/ocr/out
```

## Docker Example

As Dockerfile uses **[uv](https://docs.astral.sh/uv/)** for managing the package so you can build and run the docker container using the following commands.

```bash
docker build -t bhaashaocr .
docker run -d --rm --gpus all \
	--name bhaashaocr-container
	-v <path_to_model_folder>:/model:ro \
	-v <path_to_folder_containing_input_output>:/data \
	bhaashaocr \
	uv python infer.py \
	--pretrained /model \
	# specify the actual path for input image and output dir
	--image_path /data/image.jpg \
	--out_dir /data/output
# This will run the OCR docker container in background (-d), to see the detailed logs use the following command.
docker logs -f bhaashaocr-container
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
