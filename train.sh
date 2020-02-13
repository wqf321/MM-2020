for l in 0.001
do
    for w in 0.1 0.01 0.001 0.0001 0.00001
    do
      for d in 0 0.2 0.4 0.6 0.8
      do
             python train.py --weight_decay=$w --l_r=$l --dropout=$d
      done
    done
done