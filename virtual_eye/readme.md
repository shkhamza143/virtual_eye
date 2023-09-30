PYTHON 3.8
---------------------------------------------
## Installation

```BibTeX
git clone git@github.com:shkhamza143/virtual_eye.git
cd virtal_eye
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e .
cd ..
pip install -r requirements.txt
```


### Model Zoo
download the model from [Google Drive]()

## Getting Started

Run the Following Script
```BibTeX
python inference.py -s <source images> -t <target_images_list>
```
## Citing Detectron2

If you use virtual eye in your research or wish to refer to the baseline results published in the [Google Drive](), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Hamza Naeem},
  title =        {virtual eye},
  howpublished = {\url{https://github.com/shkhamza143/virtual_eye}},
  year =         {2023}
}
```
