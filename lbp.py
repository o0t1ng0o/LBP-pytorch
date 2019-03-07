import torch
def fast_lbp(img, device, padding_type): 
    # img: the input image
    # device : which gpu to use
    # padding_type : 0 for the zero padding, 1 for the mirror padding
    width = len(img[0])
    height = len(img)
    
    w_zeros = torch.zeros(1,width).type(torch.cuda.ByteTensor)
    h_zeros = torch.zeros(height,1).type(torch.cuda.ByteTensor)
    
    w_zeros_s = torch.zeros(1,width-1).type(torch.cuda.ByteTensor)
    h_zeros_s = torch.zeros(height-1,1).type(torch.cuda.ByteTensor)
    
    w_top_right_cut = img[0][0:width-1].view(1,width-1)
    w_top_left_cut = img[0][1:width].view(1,width-1)
    
    w_bottom_right_cut = img[height-1][0:width-1].view(1,width-1)
    w_bottom_left_cut = img[height-1][1:width].view(1,width-1)
    
    top_line = img[0:1]
    bottom_line = img[height-1:height]
    left_line = torch.split(img,1,1)[0]
    _, right_line = torch.split(img,255,1)
    if padding_type == 0:
        w_top_right_cut = w_zeros_s
        w_top_left_cut = w_zeros_s
        
        w_bottom_right_cut = w_zeros_s
        w_bottom_left_cut = w_zeros_s
        
        top_line = w_zeros
        bottom_line = w_zeros
        left_line = h_zeros
        right_line = h_zeros
    
    bottom_cut = img[0:height-1].clone()
    top_cut = img[1:height].clone()
    left_cut = torch.cat(torch.split(img, 1, 1)[1:width],1)
    right_cut, _ = torch.split(img, 255, 1)
    top_left_cut = left_cut[1: height].clone()
    top_right_cut = right_cut[1: height].clone()
    bottom_right_cut = right_cut[0:height-1].clone()
    bottom_left_cut = left_cut[0:height-1].clone()
    
    top_pad = torch.cat([top_line, bottom_cut], 0)
    bottom_pad = torch.cat([top_cut, bottom_line], 0)
    left_pad = torch.cat([left_line, right_cut], 1)
    right_pad = torch.cat([left_cut, right_line], 1)
    top_left_pad = torch.cat([w_top_right_cut, bottom_right_cut], 0)
    top_left_pad = torch.cat([left_line, top_left_pad], 1)
    top_right_pad = torch.cat([w_top_left_cut, bottom_left_cut], 0)
    top_right_pad = torch.cat([top_right_pad, right_line], 1)
    bottom_right_pad = torch.cat([top_left_cut, w_bottom_left_cut], 0)
    bottom_right_pad = torch.cat([bottom_right_pad, right_line], 1)
    bottom_left_pad = torch.cat([top_right_cut, w_bottom_right_cut], 0)
    bottom_left_pad = torch.cat([left_line, bottom_left_pad], 1)
    
    top_left = ( top_left_pad >= img)
    top = ( top_pad >= img)
    top_right = (top_right_pad >= img)
    right = (right_pad >= img)
    bottom_right = ( bottom_right_pad >= img)
    bottom = (bottom_pad >= img)
    bottom_left = (bottom_left_pad >= img)
    left = (left_pad >= img)
    
    sum = top_left * 1 + top * 2 + top_right * 4 + right * 8 + bottom_right* 16+ bottom * 32+ bottom_left * 64+ left * 128
    
    return sum
