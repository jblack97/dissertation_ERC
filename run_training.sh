MODEL_NAME = $1
echo $MODEL_NAME.bin
! python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME \
--train_batch_size_range 4 --learning_rate_range 6e-6 --hidden_dropout_prob_range 0.05 0.1 \
 --attention_probs_dropout_prob_range 0.05 0.1 --score_json_path scores.json --do_train \
 --do_eval --encoder_type BERT  --data_dir /EMORYNLP \
  --data_name EmoryNLP   --vocab_file pytorch_models/vocab.txt \
    --config_file pytorch_models/bert_config.json  \
     --init_checkpoint pytorch_models/$MODEL_NAME.bin \
         --output_dir pytorch_models/run_checkpoints  
