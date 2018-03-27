source activate tensorflow

python train_tripletloss.py \
--data_dir ../train_2 \
--batch_size 1500 \
--people_per_batch 300 \
--waves_per_person 10 \
--learning_rate 0.0001 \
--weight_decay 0.00001 \
--nrof_preprocess_threads 30