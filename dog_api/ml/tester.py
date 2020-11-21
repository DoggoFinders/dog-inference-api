import os

import torch
import torch.nn.functional as F
import pathlib
from PIL import Image
from torchvision.transforms import transforms

from .inception import inception_v3_bap
from .utils import mask2bbox


def load_image(img_data, loader):
    """load image, returns cuda tensor"""
    image = Image.open(img_data).convert("RGB")
    image = loader(image).float()
    return image.cuda()


def trim_modules(state_dict):
    return {k[len('module.'):]: v for k, v in state_dict.items()}


IMG_SIZE = 512
INPUT_SIZE = 448

ALL_LABELS = [
    'Chihuahua',
    'Japanese_spaniel',
    'Maltese_dog',
    'Pekinese',
    'Shih-Tzu',
    'Blenheim_spaniel',
    'papillon',
    'toy_terrier',
    'Rhodesian_ridgeback',
    'Afghan_hound',
    'basset',
    'beagle',
    'bloodhound',
    'bluetick',
    'black-and-tan_coonhound',
    'Walker_hound',
    'English_foxhound',
    'redbone',
    'borzoi',
    'Irish_wolfhound',
    'Italian_greyhound',
    'whippet',
    'Ibizan_hound',
    'Norwegian_elkhound',
    'otterhound',
    'Saluki',
    'Scottish_deerhound',
    'Weimaraner',
    'Staffordshire_bullterrier',
    'American_Staffordshire_terrier',
    'Bedlington_terrier',
    'Border_terrier',
    'Kerry_blue_terrier',
    'Irish_terrier',
    'Norfolk_terrier',
    'Norwich_terrier',
    'Yorkshire_terrier',
    'wire-haired_fox_terrier',
    'Lakeland_terrier',
    'Sealyham_terrier',
    'Airedale',
    'cairn',
    'Australian_terrier',
    'Dandie_Dinmont',
    'Boston_bull',
    'miniature_schnauzer',
    'giant_schnauzer',
    'standard_schnauzer',
    'Scotch_terrier',
    'Tibetan_terrier',
    'silky_terrier',
    'soft-coated_wheaten_terrier',
    'West_Highland_white_terrier',
    'Lhasa',
    'flat-coated_retriever',
    'curly-coated_retriever',
    'golden_retriever',
    'Labrador_retriever',
    'Chesapeake_Bay_retriever',
    'German_short-haired_pointer',
    'vizsla',
    'English_setter',
    'Irish_setter',
    'Gordon_setter',
    'Brittany_spaniel',
    'clumber',
    'English_springer',
    'Welsh_springer_spaniel',
    'cocker_spaniel',
    'Sussex_spaniel',
    'Irish_water_spaniel',
    'kuvasz',
    'schipperke',
    'groenendael',
    'malinois',
    'briard',
    'kelpie',
    'komondor',
    'Old_English_sheepdog',
    'Shetland_sheepdog',
    'collie',
    'Border_collie',
    'Bouvier_des_Flandres',
    'Rottweiler',
    'German_shepherd',
    'Doberman',
    'miniature_pinscher',
    'Greater_Swiss_Mountain_dog',
    'Bernese_mountain_dog',
    'Appenzeller',
    'EntleBucher',
    'boxer',
    'bull_mastiff',
    'Tibetan_mastiff',
    'French_bulldog',
    'Great_Dane',
    'Saint_Bernard',
    'Eskimo_dog',
    'malamute',
    'Siberian_husky',
    'affenpinscher',
    'basenji',
    'pug',
    'Leonberg',
    'Newfoundland',
    'Great_Pyrenees',
    'Samoyed',
    'Pomeranian',
    'chow',
    'keeshond',
    'Brabancon_griffon',
    'Pembroke',
    'Cardigan',
    'toy_poodle',
    'miniature_poodle',
    'standard_poodle',
    'Mexican_hairless',
    'dingo',
    'dhole',
    'African_hunting_dog'
]


def _load_model():
    net = inception_v3_bap(pretrained=True, aux_logits=False)

    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=120)
    net.fc_new = new_linear

    # load checkpoint
    net = net.cuda()

    local_path = pathlib.Path(__file__).parent.parent.parent / "dog.pth.tar"
    pretrained_model = torch.load(
        os.getenv("DOG_MODEL_PATH", local_path))
    net.load_state_dict(trim_modules(pretrained_model['state_dict']))
    return net


MODEL = _load_model()


# Based on code adapted from https://arxiv.org/pdf/1901.09891.pdf
# See better before looking closer: weakly supervised data augmentation

class ImageTester:

    def test(self, img_data):
        transform_test = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = load_image(img_data, transform_test)
        pathlib.Path()

        # switch to evaluate mode
        MODEL.eval()
        with torch.no_grad():
            # forward
            test_img = img.unsqueeze(0)
            attention_maps, _, output1 = MODEL(test_img)
            refined_input = mask2bbox(attention_maps, test_img)
            _, _, output2 = MODEL(refined_input)
            output = (F.softmax(output1, dim=-1) + F.softmax(output2, dim=-1)) / 2
            values, indices = output.topk(5)
            human_readable = [(int(i), ALL_LABELS[int(i.item())]) for i in indices.flatten()]
            flattened_output = (i.item() for i in values.flatten())
            results = []
            for proba, labels in zip(flattened_output, human_readable):
                results.append({"probability": proba, "label": labels[0], "human_label": labels[1]})
            return results
