MODEL_NAME = $1
echo $MODEL_NAME.bin
! python hyper_param_search.py --max_examples 100000 --model_name MODEL_NAME \
--train_batch_size_range 4 --learning_rate_range 6e-6 --hidden_dropout_prob_range 0.05 0.1 \
 --attention_probs_dropout_prob_range 0.05 0.1 --score_json_path scores.json --do_train \
 --do_eval --encoder_type BERT  --data_dir /EMORYNLP \
  --data_name EmoryNLP   --vocab_file pytorch_models/vocab.txt \
    --config_file pytorch_models/bert_config.json  \
     --init_checkpoint pytorch_models/$MODEL_NAME.bin \
       --max_seq_length 512   --train_batch_size 12   --learning_rate 1e-5   --num_train_epochs 7.0 \
         --output_dir pytorch_models/run_checkpoints   --gradient_accumulation_steps 1 
