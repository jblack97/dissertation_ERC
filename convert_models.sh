MODEL_NAME="${1}" NUM_EPOCHS="${2}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS}\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_${MODEL_NAME}_${NUM_EPOCHS}.bin
NUM_EPOCHS="${3}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS}\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_${MODEL_NAME}_${NUM_EPOCHS}.bin
NUM_EPOCHS="${4}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS}\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_${MODEL_NAME}_${NUM_EPOCHS}.bin
NUM_EPOCHS="${5}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS} \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_${MODEL_NAME}_${NUM_EPOCHS}.bin
NUM_EPOCHS="${6}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS}\
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_${MODEL_NAME}_${NUM_EPOCHS}.bin
NUM_EPOCHS="${7}"
python convert_model.py --tf_checkpoint /tf_models/BERT_${MODEL_NAME}_epoch_0_step_${NUM_EPOCHS}/bert_model_epoch_0_step_${NUM_EPOCHS}.ckpt-${NUM_EPOCHS} \
--config_file tf_models/bert_config.json --pytorch_dump_path pytorch_models/pytorch_model_${MODEL_NAME}_${NUM_EPOCHS}.bin
