python convert_model.py --tf_checkpoint /tf_models/BERT_SND_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-80000\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_SND_80000.bin

python convert_model.py --tf_checkpoint /tf_models/BERT_RUR_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-80000\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_RUR_80000.bin

python convert_model.py --tf_checkpoint /tf_models/BERT_MSUR_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-80000\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_MSUR_80000.bin

python convert_model.py --tf_checkpoint /tf_models/BERT_PCD_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_PCD_80000.bin

python convert_model.py --tf_checkpoint /tf_models/BERT_ISS_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-80000\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_ISS_80000.bin

python convert_model.py --tf_checkpoint /tf_models/BERT_NSP_MLM_epoch_0_step_80000/bert_model_epoch_0_step_80000.ckpt-80000 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_NSP_MLM_80000.bin
