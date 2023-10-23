import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import torch
import numpy as np
import json

#CONFIG = "configs/coco/dino_4scale_internimage_l_1x_coco_layer_wise_lr.py"
CONFIG = "configs/coco/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.py"
#CKPT = "checkpoint_dir/det/dino_4scale_internimage_l_1x_coco_layer_wise_lr.pth"
CKPT = "checkpoint_dir/det/dino_4scale_internimage_l_1x_coco_0.1x_backbone_lr.pth"

SCORE_TRESHOLD = 0.3
ASYNC_TEST = 'store_true'

PALETTES = ['coco', 'voc', 'citys', 'random']

OUT = "demo"
JSON_OUT = "Json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_get_bboxes(model, img, treshold):
    b_boxes = inference_detector(model, img)

    new_b_boxes = []
    for id in b_boxes:
        saved_bboxes = []
        for bbox in id:
            if bbox[-1] > treshold:
                saved_bboxes.append(bbox)

        if saved_bboxes:
            new_b_boxes.append(np.array(saved_bboxes))
        else:
            new_b_boxes.append(id)

    print(len(b_boxes))
    print(len(new_b_boxes))

    mmcv.mkdir_or_exist(OUT)
    out_file = osp.join(OUT, osp.basename(args.img).split(".")[0] + "_extracted_bboxes.jpg")
    # show the results
    model.show_result(
        args.img,
        new_b_boxes,
        score_thr=SCORE_TRESHOLD,
        show=False,
        bbox_color=PALETTES[0], #Index of the corresponding dataset
        text_color=(200, 200, 200),
        out_file=out_file
    )
    print(f">>> Test image saved at : {out_file}")

def get_bboxes(model, img, treshold, show_im=False):
    """
    Return : Dictionnary :
    dict[str : classes] = [
        {
            "bbox" : {
                        "x" : x value,
                        "y" : y value,
                        "width" : width in pixel,
                        "height" : height in pixel
                    },
            "confidence" : confidence
        },
        {
            "bbox" : {
                        "X" : x2 value,
                        "Y" : y2 value,
                        "Width" : width in pixel
                        "Height" : height in pixel
                    },
            "confidence" : confidence
        },
        ...
    ]
    """
    dico = {}
    labels = model.CLASSES

    b_boxes = inference_detector(model, img)

    new_b_boxes = []
    for i, id in enumerate(b_boxes):
        saved_bboxes = []

        for bbox in id:
            dico_temp = {}
            if bbox[-1] > treshold:
                D = {}
                D["x"] = bbox.tolist()[0]
                D["y"] = bbox.tolist()[1]
                D["x2"] = bbox.tolist()[2]
                D["y2"] = bbox.tolist()[3]

                dico_temp["bbox"] = D
                dico_temp["confidence"] = bbox.tolist()[-1]
                saved_bboxes.append(dico_temp) #Removing prediction indices

        if saved_bboxes:
            dico[labels[i]] = saved_bboxes
            new_b_boxes.append(saved_bboxes)

    print(f">>> Total classes : {len(b_boxes)}")
    print(f">>> Detected classes : {len(new_b_boxes)}")

    if show_im:
        mmcv.mkdir_or_exist(OUT)
        out_file = osp.join(OUT, osp.basename(args.img))
        # show the results
        model.show_result(
            args.img,
            b_boxes,
            score_thr=SCORE_TRESHOLD,
            show=False,
            bbox_color=PALETTES[0], #Index of the corresponding dataset
            text_color=(200, 200, 200),
            out_file=out_file
        )
        print(f">>> Image saved at : {out_file}")

    return dico

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(CONFIG, CKPT, device=DEVICE)

    #Uncomment to save the result of construction and extraction of b_boxes test
    #test_get_bboxes(model, args.img, SCORE_TRESHOLD)

    #Get the dictionnary of classes containing bboxes
    b_boxes = get_bboxes(model, args.img, SCORE_TRESHOLD, show_im=True)

    mmcv.mkdir_or_exist(JSON_OUT)
    out_json = osp.join(JSON_OUT, osp.basename(args.img).split(".")[0] + ".json")

    with open(out_json, "w") as json_file:
        json.dump(b_boxes , json_file)
        print(f">>> Json written at : {out_json}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
