python train.py --batch_sz 4 --gradient_accumulation_steps 40 \
 --savedir checkpoints/ --name bert_1 \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model bert  \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1