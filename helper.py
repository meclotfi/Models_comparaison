from PIL import Image
import os
import numpy as np
import torch
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool

def load_images_from_folder(folder,maxs=8):
    images = []
    i=0
    for filename in os.listdir(folder):
        img =os.path.join(folder,filename)
        if img is not None:
            images.append(img)
            i+=1
        if i==maxs: 
            break
    return images


configs={
    "Swin":"./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    "Swin_fp-16_mss-crop":"./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
    "Detr-R50":"./configs/detr/detr_r50_8x2_150e_coco.py",
    "Def-DETR-R50":"./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
    "Def-DETR-R50+iterative_bbox":"./configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py",
    "Def-DETR-R50++two_stage":"./configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py",
    "PVT-Tiny":"./configs/pvt/retinanet_pvt-t_fpn_1x_coco.py",
    "PVTv2-B0":"./configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py",
    "M_rcnn":"./configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py",
    "Yolov3":"./configs/yolo/yolov3_d53_mstrain-608_273e_coco.py"
}
def Write_data_to_csv(file_path,model,conf,Bs,val_mean=None,val_var=None,mem=None):
   S='{model},{conf},{BS},{t_mean},{t_var},{mem}\n'.format(model=model,conf=conf, BS=Bs,t_mean=val_mean,t_var=val_var,mem=mem)
   with open(file_path, 'a+') as f:
      f.write(S)
   print(S)

def Prepare_data(imgs,model):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
    return data

def inference_det(model,data,is_batch):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    
    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
