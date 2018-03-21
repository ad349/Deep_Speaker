source activate tensorflow


python train_tripletloss.py \
--data_dir ../train_2 \
--learning_rate 0.0001 \
--waves_per_person 10 \
--people_per_batch 150 \
--batch_size 30 \
--max_nrof_epochs 100 \
--learning_rate_decay_factor 0.9 \
--learning_rate_decay_epochs 2 \
--keep_probability 0.1 \
--nrof_preprocess_threads 30 \
--epoch_size 5000 \
--pretrained_model "./models/bioid/20180320-150128/model-20180320-150128.ckpt-6973"
