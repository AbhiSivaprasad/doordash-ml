for filename in processed/*; do
	write_path=preds/$(basename $filename)
	echo $write_path
	python3 -m scripts.batch_predict --test_path $filename --write_path $write_path --predict_batch_size 512 --model_type huggingface --model_dir local/models/nlp-best/ --taxonomy taxonomy-doordash:latest --strategy complete
done

