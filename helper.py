from PIL import Image
import os

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
    "Detr-R50":"./configs/detr/detr_r50_8x2_150e_coco.py",
    "Def-DETR-R50":"./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
    "Def-DETR-R50+iterative_bbox":"./configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py",
    "Def-DETR-R50++two_stage":"./configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py",
    "PVT-Tiny":"./configs/pvt/retinanet_pvt-t_fpn_1x_coco.py",
    "PVTv2-B0":"./configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py"
}
def Write_data_to_csv(file_path,model,conf,Bs,val_mean=None,val_var=None,mem=None):
   S='{model},{conf},{BS},{t_mean},{t_var},{mem}\n'.format(model=model,conf=con, BS=Bs,t_mean=val_mean,t_var=val_var,mem=mem)
   with open(file_path, 'a+') as f:
      f.write(S)
   print(S)