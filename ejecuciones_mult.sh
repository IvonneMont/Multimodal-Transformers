python train.py --batch_sz 1 --gradient_accumulation_steps 8 \
 --savedir checkpoints/ --name mult_1 --type train \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mult --num_image_embeds 3 --num_heads 1\
 --patience 5 --dropout 0.1 --lr 0.5 --warmup 0.1 --max_epochs 30 --seed 2