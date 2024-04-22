## StarGAN

**\*\*\*\*\* Project based on https://github.com/yunjey/stargan/tree/master \*\*\*\*\***

## Dataset formats
CelebA-format datasets:
  - images/ folder, containing arbitrary named images
  - list_attr.txt:
    - number of images on the first line
    - names of labels on the second line
    - for each image, a line describing the attributes it possesses

<br>

RaFD-format datasets:
  - train/ and test/ folders, each with the following structure:
    - arbitrary named folders, each folder naming a categorical attribute
    - inside each folder, arbitrary named images

<br>

Visualization datasets (used for tracking progress / debugging):
  - list of arbitrary named images

<br>

Preffered image size: 178*218.

Images that have a different spatial shape can be resized using [resize_v1.ipynb](https://github.com/mihai145/iava_proj/blob/main/resize_v1.ipynb) (without face detection) or [resize_v2.ipynb](https://github.com/mihai145/iava_proj/blob/main/resize_v2.ipynb) (with face detection).