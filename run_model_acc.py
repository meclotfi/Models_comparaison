from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import sys
from helper import configs,load_images_from_folder,Write_data_to_csv,Prepare_data,inference_det
import time
import os
import torch
import warnings
warnings.filterwarnings("ignore")

config=configs[sys.argv[1]]
Bs=int(sys.argv[2])
dev=int(sys.argv[3])

imgs=load_images_from_folder("data",maxs=Bs)
model = init_detector(config, device=dev)

is_batch=False
if Bs>1:
    is_batch=True
data=Prepare_data(model=model,imgs=imgs)

stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
m1=int(out2[ind_used-2])

result = inference_det(model,imgs,is_batch=is_batch)

stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
mem=int(out2[ind_used-2])-m1


Results=[]
          # calculating latency by averaging over num_trials 
for i in range(5):
    start=time.time()
    res = inference_det(model,imgs,is_batch=is_batch)
    end=time.time()
    Results.append(end-start)
          
# Saving the results 
r=torch.Tensor(Results)

Write_data_to_csv("/content/gdrive/My Drive/Attention_eff/Model-all-cpu.csv",model=sys.argv[1],conf=config,Bs=Bs,val_mean=r.mean().item(),val_var=r.var().item(),mem=mem)