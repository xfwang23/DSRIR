
iteration = 400000  # total iteration
batch_size = [16, 14, 12, 8]

# iter_ = [128, 240, 336, 400]  # K, in which iteration switch to corresponding dataloader
iterat = [128, 112, 96, 64]  # number of iteration for different dataloader stage.


def cal_training_epochs(num_img):
    total = 0
    for i in range(len(batch_size)):
        iter_per_epoch = num_img/batch_size[i]
        epoch_per_stage = (iterat[i] * 1000)//iter_per_epoch
        total += epoch_per_stage
        print(total)
    # print('epoch: ', total)


# 4744+400+5000+5000 = 15144
# All-in-one
cal_training_epochs(num_img=15144)

# Denoising
# cal_training_epochs(num_img=4744+500)

# # Deraining
# cal_training_epochs(num_img=5000)

# # Dehazing
# cal_training_epochs(num_img=5000)

