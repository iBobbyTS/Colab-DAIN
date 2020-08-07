import time
import os
from torch.autograd import Variable
import torch
import numpy as np
import networks
import warnings

res = 1000
all_start_time = time.time()

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True

model = networks.__dict__['DAIN_slowmotion'](
    channel=3,
    filter_size=4,
    timestep=0.25,
    training=False)

model = model.cuda()

model_path = './model_weights/best.pth'
if not os.path.exists(model_path):
    print("*****************************************************************")
    print("**** We couldn't load any trained weights ***********************")
    print("*****************************************************************")
    exit(1)

pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
# 4. release the pretrained dict for saving memory
pretrained_dict = []

model = model.eval()  # deploy mode


timestep = 0.1
time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]



# frame_count_len = len(str(final_frame))

torch.set_grad_enabled(False)

# we want to have input_frame between (start_frame-1) and (end_frame-2)
# this is because at each step we read (frame) and (frame+1)
# so the last iteration will actuall be (end_frame-1) and (end_frame)

while True:
    # input_frame += 1

    start_time = time.time()

    X0 = torch.from_numpy(np.random.randint(0,255,(3, res, res)).astype("float32") / 255.0).type(
        torch.cuda.FloatTensor)
    X1 = torch.from_numpy(np.random.randint(0,255,(3, res, res)).astype("float32") / 255.0).type(
        torch.cuda.FloatTensor)

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channels = X0.size(0)

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft = int((intWidth_pad - intWidth) / 2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intPaddingLeft = 32
        intPaddingRight = 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

    X0 = Variable(torch.unsqueeze(X0, 0))
    X1 = Variable(torch.unsqueeze(X1, 0))
    X0 = pader(X0)
    X1 = pader(X1)


    X0 = X0.cuda()
    X1 = X1.cuda()

    y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
    y_ = y_s[1]


    X0 = X0.data.cpu().numpy()
    if not isinstance(y_, list):
        y_ = y_.data.cpu().numpy()
    else:
        y_ = [item.data.cpu().numpy() for item in y_]
    offset = [offset_i.data.cpu().numpy() for offset_i in offset]
    filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
    X1 = X1.data.cpu().numpy()


    X0 = np.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                              intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
    y_ = [np.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                               intPaddingLeft:intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
    offset = [
        np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                     (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter] if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                              intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

    interpolated_frame_number = 0
    for item, time_offset in zip(y_, time_offsets):
        interpolated_frame_number += 1
        

    end_time = time.time()
    time_spent = end_time - start_time
    all_time_spent = time.time()-all_start_time
    m, s = divmod(all_time_spent, 60)
    h, m = divmod(m, 60)
    estimated_time_left = "%d:%02d:%02d" % (h, m, s)
    print(f"******  | Time spent: {round(time_spent, 2)}s | Total: {estimated_time_left} ******************")

