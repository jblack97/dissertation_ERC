MODEL_NAME="${1}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints

MODEL_NAME="${2}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints

MODEL_NAME="${3}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints

MODEL_NAME="${4}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints

MODEL_NAME="${5}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints

MODEL_NAME="${6}" MODEL_PATH="pytorch_model_${MODEL_NAME}.bin"
echo $MODEL_PATH
python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME --train_batch_size_range 4 --learning_rate_range  5e-6 8e-6 --hidden_dropout_prob_range 0.05  --attention_probs_dropout_prob_range 0.05  --score_json_path scores.json --do_train --do_eval --encoder_type BERT  --data_dir /EmoryNLP --data_name EmoryNLP   --vocab_file /pytorch_models/vocab.txt --config_file /pytorch_models/bert_config.json --init_checkpoint /pytorch_models/$MODEL_PATH --output_dir /pytorch_models/run_checkpoints
