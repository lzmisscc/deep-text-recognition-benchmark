import uuid
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

compose = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.88, 0.18)
])



class Predict:
    def __init__(self, ):
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        
        model.eval()
        self.converter = converter
        self.model = model

    def run(self, im):
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in zip([0], [0]):
                # im = Image.open(image_tensors)
                im = pre_pre(im, opt)
                image_tensors = compose(im)
                image_tensors = torch.unsqueeze(image_tensors, 0).float()
                batch_size = image_tensors.size(0)
                image_path_list = [image_path_list, ]
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)



                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    pred = index_decode(pred)
                    return pred, confidence_score


def pre_pre(img, opt):
    import math
    image = img
    w, h = image.size
    # name = uuid.uuid4().hex
    # image.save(f"VIS/{name}-gt.jpg")
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) <= opt.imgW:
        resized_w = opt.imgW
        resized_image = image.resize(
            (resized_w, opt.imgH), Image.BICUBIC)

        new_PAD = Image.new(size=(opt.imgW, opt.imgH), color=(255),mode='L')
        new_PAD.paste(resized_image, (0,0))
        img = new_PAD  
        # img.save(f"VIS/{name}.jpg") 
        return img
    return image

import re
def index_decode(index_encode):
    # 解码部分
    res = re.sub("\^{(.*?)}", r"<sup>\1</sup>", index_encode)
    res = re.sub("_{(.*?)}", r"<sub>\1</sub>", res)
    res = re.split("(.{0,1}[卐♡♀]{0,3})", res)
    res = list(filter(lambda x: x, res))


    def trans(data, split='卐', start='<b>', end='</b>'):
        res = data

        tmp_res = []
        while res:
            tmp = res.pop(0)
            tmp_ = []
            if split in tmp:
                tmp_ = [start + tmp.replace(split, ""), ]
                if res:
                    tmp = res.pop(0)
                    flag = 0
                    while split in tmp:
                        tmp_.append(tmp.replace(split, ""))
                        if res:
                            tmp = res.pop(0)
                        else:
                            flag = 1
                            break

                    tmp_[-1] += end
                    if not flag:
                        tmp_.append(tmp)
                    tmp_res += tmp_
                else:
                    tmp_res.append(start + tmp.replace(split, "") + end,)
            else:
                tmp_res.append(tmp)
        return tmp_res

    res = trans(res, '♡', '<i>', '</i>')
    res = trans(res, '卐', '<b>', '</b>')
    res = trans(res, '♀', '<strike>', '</stirke>')


    return ''.join(res)

if __name__ == '__main__':
    from CONFIG2TRAIN.config import opt
    # from CONFIG2TRAIN.seq2seq import opt
    # from CONFIG2TRAIN.seq2seq_attn import opt
    # from CONFIG2TRAIN.attention import opt

    from eval import Ev
    import time
    import os
    import os.path as osp

    ev = Ev()
    run = Predict().run
    # opt.saved_model = "saved_models/Weight/best_accuracy.pth"
    opt.batch_size = 1
    opt.imgW = 32
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    table_ocr_txt_path = "../table_ocr/filter_val.txt"
    with open(table_ocr_txt_path, "r") as f:
        gt_lines = f.readlines()
    for index, line in enumerate(gt_lines):
        name, value = line.strip("\n").split("\t")
        im = Image.open(osp.join("../table_ocr/data/val", name))
        start = time.time()
        pre,conf = run(im)
        ev.count(value, pre)
        print(f"{value}\t{pre}\t{value==pre}")
        print(f"{time.time()-start:.2f}\t{ev.socre()}")

