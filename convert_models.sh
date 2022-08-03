

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_SND_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_SND.bin

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_RUR_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_RUR.bin

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_MSUR_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_MSUR.bin

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_PCD_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_PCD.bin

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_ISS_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_ISS.bin

! python /dissertation_ERC/convert_model.py --tf_checkpoint /ERC_training/tf_models/BERT_NSP_MLM_epoch_0_step_194828/bert_model_epoch_0_step_194828.ckpt-194828 \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_NSP_MLM.bin
