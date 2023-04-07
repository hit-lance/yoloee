import torch


def set_grid(input_size, anchor_size, device):
    w, h = input_size, input_size
    # generate grid cells
    ws, hs = w // 32, h // 32
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    grid_cell = grid_xy.view(1, hs * ws, 1, 2).to(device)

    # generate anchor_wh tensor
    all_anchor_wh = torch.tensor(anchor_size).repeat(hs * ws, 1, 1).unsqueeze(0).to(device)

    return grid_cell, all_anchor_wh


def decode_boxes(txtytwth_pred, grid_cell, all_anchor_wh):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
    """
    # txtytwth -> cxcywh
    B = txtytwth_pred.size()[0]
    xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + grid_cell
    wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * all_anchor_wh
    # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
    xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * 32

    # cxcywh -> x1y1x2y2
    x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
    x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
    x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

    return x1y1x2y2_pred

