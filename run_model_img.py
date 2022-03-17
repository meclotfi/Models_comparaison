from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import sys
import time
import os
import torch
import warnings
from mmcv import Config
warnings.filterwarnings("ignore")

config=sys.argv[1]
img=sys.argv[2]
dev=sys.argv[3]
cfg = Config.fromfile(config)
#imgs=load_images_from_folder("data",maxs=Bs)
model = init_detector(config,device=dev)

stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
m1=int(out2[ind_used-2])

result = inference_detector(model, [img])

stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
mem=int(out2[ind_used-2])-m1


Results=[]
          # calculating latency by averaging over num_trials 
for i in range(5):
    start=time.time()
    res = inference_detector(model, [img])
    end=time.time()
    Results.append(end-start)
          
# Saving the results 
r=torch.Tensor(Results)

print("Modl: "+cfg.model.type + "  Mem consumption: "+str(mem)+"  Inference time: "+str(r.mean().item()))