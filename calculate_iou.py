import torch

# Example ground truth box (x_center, y_center, width, height)
gt_box = torch.tensor([0.5, 0.5, 0.4, 0.4])

# Example anchors (width, height)
anchors = torch.tensor([[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]])

# Calculate IoU between ground truth box and anchors
def calculate_iou(box1, box2):
    inter = torch.min(box1[2:], box2[2:]) - torch.max(box1[:2], box2[:2])
    inter = torch.clamp(inter, min=0)
    inter_area = inter[0] * inter[1]
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

ious = [calculate_iou(gt_box, torch.cat((torch.tensor([0.5, 0.5]), anchor))) for anchor in anchors]

# Assign anchor with highest IoU
best_anchor_idx = torch.argmax(torch.tensor(ious))
print(f"Best anchor index: {best_anchor_idx}, IoU: {ious[best_anchor_idx]}")