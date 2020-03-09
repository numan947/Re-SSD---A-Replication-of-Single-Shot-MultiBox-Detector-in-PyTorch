from commons import *
import torch


def perform_nms(priors_cxcy, n_classes, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)

    predicted_scores = torch.softmax(predicted_scores, dim=-1)

    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()


    assert n_priors == predicted_scores.size(1) == predicted_locs.size(1)

    for i in range(batch_size):
        decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))

        image_boxes = list()
        image_labels = list()
        image_scores = list()

        for c in range(1, n_classes):
            class_scores = predicted_scores[i][:, c]
            score_above_min_score = class_scores>min_score
            n_above_min_score = score_above_min_score.sum().item()

            if n_above_min_score == 0:
                continue

            class_scores = class_scores[score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]

            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
            class_decoded_locs = class_decoded_locs[sort_ind]

            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

            suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)

            for box in range(class_decoded_locs.size(0)):
                if suppress[box]:
                    continue
                suppress = torch.max(suppress, overlap[box]>max_overlap)
                suppress[box] = False

            image_boxes.append(class_decoded_locs[~suppress])
            image_labels.append(torch.LongTensor([c]*(~suppress).sum().item()).to(device))
            image_scores.append(class_scores[~suppress])

        if len(image_labels) == 0:
            image_boxes.append(torch.FloatTensor([[0.0,0.0,1.0,1.0]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.0]).to(device))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        n_objects = image_scores.size(0)

        if n_objects>top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]
            image_boxes = image_boxes[sort_ind][:top_k]
            image_labels = image_labels[sort_ind][:top_k]

        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores