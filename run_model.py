from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import sys
from helper import configs,load_images_from_folder
import time

config=configs[sys.argv[1]]
Bs=int(sys.argv[2])


imgs=load_images_from_folder("data",maxs=Bs)
model = init_detector(config, device='cpu')

start=time.time()
result = inference_detector(model, imgs)
end=time.time()

print(start-end)