/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 4 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name mmbt_mm_vqgan_1 --type test --modality ti\
 --data_path data_sets/proc/ \
 --task mmimdb --task_type multilabel \
 --model mmbt --num_image_embeds 256 --img_hidden_sz 1024 --encoder VQGAN --poster features\
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1