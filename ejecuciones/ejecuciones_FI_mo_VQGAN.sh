/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 32 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name FI_MO_1_VQGAN --type train \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model FastImage  --num_features_img 1024 --size_sequence_img 256 --encoder VQGAN \
 --patience 5 --dropout 0.5 --lr 0.5 --warmup 0.1 --max_epochs 100 --seed 1
 
 
 
 