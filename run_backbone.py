from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import sys
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.models.builder import build_backbone
from mmcv import Config
import time
import os
import torch
import warnings
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate
warnings.filterwarnings("ignore")

config_file=sys.argv[1]
img_file=sys.argv[2]

cfg = Config.fromfile(config_file)
back = build_backbone(cfg.model.backbone)

cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
test_pipeline = Compose(cfg.data.test.pipeline)
back.eval()


stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
m1=int(out2[ind_used-2])

data = dict(img_info=dict(filename=img_file), img_prefix=None)
datas=[]
data=test_pipeline(data)
datas.append(data)
data = collate(datas,samples_per_gpu=1)
result = back.forward(data['img'][0].data[0])

stream = os.popen('vmstat -s')
out2 = stream.read().split()
ind_used=out2.index('used')
mem=int(out2[ind_used-2])-m1


Results=[]
          # calculating latency by averaging over num_trials 
for i in range(5):
    start=time.time()
    data = dict(img_info=dict(filename=img_file), img_prefix=None)
    datas=[]
    data=test_pipeline(data)
    datas.append(data)
    data = collate(datas,samples_per_gpu=1)
    result = back.forward(data['img'][0].data[0])
    end=time.time()
    Results.append(end-start)
          
# Saving the results 
r=torch.Tensor(Results)

print("Backbone: "+cfg.model.backbone.type + "  Mem consumption: "+str(mem)+"  Inference time: "+str(r.mean().item()))