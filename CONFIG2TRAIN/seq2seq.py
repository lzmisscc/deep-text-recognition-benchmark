from CONFIG2TRAIN.config import opt
opt.exp_name = "Seq2Seq"
opt.Prediction = "CTC"
opt.SequenceModeling = "BiLSTM"
opt.saved_model = "saved_models/Seq2Seq/best_norm_ED.pth"
# opt.saved_model = ""

# 在本次的训练中使用了随机裁剪增广旋转的增广。