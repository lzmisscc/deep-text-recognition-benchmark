from CONFIG2TRAIN.config import opt
import logging
from table_ocr.char_100_20201210 import char
opt.character = char
opt.exp_name = "char_100_max_length_50_fliter_b_i"
opt.Prediction = "CTC"
opt.SequenceModeling = None
opt.data_filtering_off = False
opt.grad_clip = 50
opt.batch_max_length = 50

# opt.saved_model = "saved_models/char_100_max_length_50_fliter_b_i/best_norm_ED.pth"
opt.saved_model = ""
opt.fliter_b_i = True
# 在本次的训练中使用了随机裁剪增广旋转的增广。
logging.warning(opt.character)
