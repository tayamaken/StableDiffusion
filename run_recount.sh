echo 'STARTING EXPERIMENT' $1 $2 $3;
python3 angles_recount.py \
	--dataset $1 \
	--learning_rate $2 \
	--max_train_steps $3;
s3cmd put --recursive outputs/$1/* s3://dkozl-object-storage/$1/;
rm -rf outputs/$1/*;
echo 'FINISHED EXPERIMENT' $1 $2 $3;
