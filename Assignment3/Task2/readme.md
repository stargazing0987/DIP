# Task2 - Combine DragGAN with Automatic Face Landmarks

## Requirements

To prepare DragGAN

```DragGAN
git clone https://github.com/XingangPan/DragGAN.git
```

Then update two files in current folder

To setup environment

```env
conda env create -f environment.yml
conda activate stylegan3
```

To download pretrained model

```
python scripts/download_model.py
```

To run DragGAN+face_alignment

```run
scripts/gui.sh
```
or
```
scripts/gui.bat
```

## Introduction

In file visualizer_drag, we draw feature points of the face.

In file.\viz\drag_widget, we add two functions: smile and thin face.

How to use the gui :

1.Click ''gen points'',and we get feature points colored green;

2.Click ''smile'' or ''thin face'', then click ''start'', and we get automatic picture processing;

3.everytime we are able to use "smile" or "thin face",click "gen points" first to get current feature points.

## Result