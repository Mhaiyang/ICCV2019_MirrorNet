"""
 @Time    : 9/15/19 10:17
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : evaluation.py
 @Function:
 
"""
from misc import *
from config import msd_testing_root, msd_results_root

# Available methods: Statistics, PSPNet, ICNet, Mask RCNN, DSS, PiCANet, RAS, R3Net, DSC, BDRAR, MirrorNet
test_method = 'DSS'

ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(msd_testing_root, "image")
MASK_DIR = os.path.join(msd_testing_root, "mask")
PREDICT_DIR = os.path.join(msd_results_root, test_method)

imglist = os.listdir(IMAGE_DIR)

print("Total {} test images".format(len(imglist)))

IOU = []
ACC = []
MAE = []
BER = []
NUM = []

for i, imgname in enumerate(imglist):

    if imgname == '1751_512x640.jpg':
        print("1751.jpg has wrong mask!")
        continue

    print("###############  {}   ###############".format(i+1))

    gt_mask = get_gt_mask(imgname, MASK_DIR)
    predict_mask_normalized = get_normalized_predict_mask(imgname, PREDICT_DIR)
    predict_mask_binary = get_binary_predict_mask(imgname, PREDICT_DIR)

    iou = compute_iou(predict_mask_binary, gt_mask)
    acc = compute_acc_mirror(predict_mask_binary, gt_mask)
    mae = compute_mae(predict_mask_normalized, gt_mask)
    ber = compute_ber(predict_mask_binary, gt_mask)

    print("iou : {}".format(iou))
    print("acc : {}".format(acc))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    IOU.append(iou)
    ACC.append(acc)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split(".")[0]
    NUM.append(num)

mean_IOU = 100 * sum(IOU) / len(IOU)
mean_ACC = sum(ACC) / len(ACC)
mean_MAE = sum(MAE) / len(MAE)
mean_BER = 100 * sum(BER) / len(BER)

print(len(IOU))
print(len(ACC))
print(len(MAE))
print(len(BER))

data_write(os.path.join(ROOT_DIR, 'excel', '%s.xlsx' % test_method), [NUM, [100*x for x in IOU], ACC, MAE, [100*x for x in BER]])

print("{}, \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}\n".format(PREDICT_DIR, "mean_IOU", mean_IOU, "mean_ACC", mean_ACC, "mean_MAE", mean_MAE, "mean_BER", mean_BER))
