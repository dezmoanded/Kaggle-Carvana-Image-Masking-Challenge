from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_1024_heng, get_unet_1920x1280

input_size = 1024

input_width = 1920
input_height = 1280

max_epochs = 50
batch_size = 6

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet_1920x1280
