from CONFIG2TRAIN.config import opt
opt.exp_name = "Attn"
opt.Prediction = "Attn"
opt.data_filtering_off = True
opt.grad_clip = 50

opt.saved_model = "saved_models/Attn/best_norm_ED.pth"
# opt.saved_model = ""

# 在本次的训练中使用了随机裁剪增广旋转的增广。