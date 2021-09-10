python train.py --batch_sz 4 --gradient_accumulation_steps 20 \
 --savedir checkpoints/ --name mmbt_mmimdb_i1 --type train --modality ti\
 --data_path data_sets/proc/ \
 --task mmimdb --task_type multilabel --freeze_txt 5 --freeze_img 3 \
 --model mmbt --num_image_embeds 3 \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1