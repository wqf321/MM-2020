for l in 0.1 0.01 0.001 0.0001 0.00001
do
    for w in 0.1 0.01 0.001 0.0001 0.00001
    do
            CUDA_VISIBLE_DEVICES=0 python train_base.py --weight_decay=$w --l_r=$l
    done
done
