

for Name in "Swin" "Detr-R50" "Def-DETR-R50" "Def-DETR-R50+iterative_bbox" "Def-DETR-R50++two_stage" "PVT-Tiny" "PVTv2-B0"
do
  for Bs in 1 4 8 16
  do
     python3 ./run_model.py $Name $Bs
      
  done
done