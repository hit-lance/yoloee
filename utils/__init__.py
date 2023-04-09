def divide(pred, num_classes=20, num_anchors=5):
    B, abC, H, W = pred.size()

    # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
    pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

    # [B, H*W*num_anchor, 1]
    conf_pred = pred[:, :, :1 * num_anchors].contiguous().view(
        B, H * W * num_anchors, 1)
    # [B, H*W, num_anchor, num_cls]
    cls_pred = pred[:, :, 1 * num_anchors:(1 + num_classes) *
                    num_anchors].contiguous().view(B, H * W * num_anchors,
                                                   num_classes)
    # [B, H*W, num_anchor, 4]
    reg_pred = pred[:, :, (1 + num_classes) * num_anchors:].contiguous().view(
        B, H * W*num_anchors, 4)

    return conf_pred, cls_pred, reg_pred
