# dog-inference-api

This API serves as a wrapper on a "special" type of inception_v3 network presented in the paper [See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/pdf/1901.09891.pdf), using some adapted pytorch code to obtain probabilities from image payloads.

The model is trained on the [Stanford dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset.
## Running

```
flask run
```

## Usage

Just POST an image at `/api/infer` content-type `multipart/form-data`. 

The image file must be named `image` in the payload.

You should get back a json with the class and top5 probabilities:

```json
[
  {
    "human_label": "cocker_spaniel",
    "label": 68,
    "probability": 0.9228276014328003
  },
  {
    "human_label": "Sussex_spaniel",
    "label": 69,
    "probability": 0.06862740963697433
  },
  {
    "human_label": "Irish_setter",
    "label": 62,
    "probability": 0.005550186149775982
  },
  {
    "human_label": "Welsh_springer_spaniel",
    "label": 67,
    "probability": 0.0014432468451559544
  },
  {
    "human_label": "Gordon_setter",
    "label": 63,
    "probability": 0.0005733264843001962
  }
]
```