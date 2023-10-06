echo 'STARTING EXPERIMENT' $1 $2 $3;
python3 TI.py \
	--save_smth_path ./outputs/trash \
	--save_root_path ./outputs/$1 \
	--dataset $1 \
	--learning_rate $2 \
	--max_train_steps $3 \
	--num_samples 2;
s3cmd sync outputs/$1/* s3://dkozl-object-storage/$1/ --recursive;
rm -rf outputs/$1/*;
rm -rf outputs/trash/*;
echo 'FINISHED EXPERIMENT' $1 $2 $3;
