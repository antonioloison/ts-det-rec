# ts-det-rec
Algorithms for traffic sign detection and recognition

## Installation

First, install the packages required in `requirements.txt`, with the following command in the main folder of the project:

```
pip install -r requirements.txt
```

## Handmade Detrec

This folder contains an implementation of the detection algorithm from the "Robust Automatic Traffic Signs Detection Using Fast Polygonal Approximation of Digital
Curves" paper. This simple implementation can be used to practice image treatment and for small robots that don't have much computation power.
The neural networks were built from scratch using numpy and are a good exercise for AI beginners.
To test this implementation, follow these commands:

```
cd handmade-detrec/recognition
python video_process.py
```


