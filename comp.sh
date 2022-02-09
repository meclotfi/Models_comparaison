
for Name in "Swin"
do
  for Bs in 1 4 8 16
  do
     python3 ./run_model.py $Name $Bs
      
  done
done