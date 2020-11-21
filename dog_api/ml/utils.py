import torch


def mask2bbox(attention_maps, input_image):
    input_tensor = input_image
    B, C, H, W = input_tensor.shape
    batch_size, num_parts, Hh, Ww = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps, size=(W, H), mode='bilinear')
    ret_imgs = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        # print(attention_map.shape)
        mask = attention_map.mean(dim=0)
        # print(type(mask))
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        # threshold = random.uniform(0.4, 0.6)
        threshold = 0.1
        max_activate = mask.max()
        min_activate = threshold * max_activate
        itemindex = torch.nonzero(mask >= min_activate)

        padding_h = int(0.05 * H)
        padding_w = int(0.05 * W)
        height_min = itemindex[:, 0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[:, 0].max() + padding_h
        width_min = itemindex[:, 1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[:, 1].max() + padding_w
        # print(height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        # print(out_img.shape)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    # print(ret_imgs.shape)
    return ret_imgs
