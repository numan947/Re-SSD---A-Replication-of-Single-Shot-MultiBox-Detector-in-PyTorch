from commons import *
from preprocess import test_image_transform
from output_processing import *
from PIL import Image, ImageDraw, ImageFont


def detect_single_image(model, original_image, min_score, max_overlap, top_k, suppress=None):
    model.eval()
    image = test_image_transform(original_image).to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = perform_nms(model.priors_cxcy, model.n_classes, predicted_locs, predicted_scores,
                                                    min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    print(det_boxes.shape)
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes*original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = det_scores[0].to('cpu').tolist()

    if det_labels == ['background']:
        return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibri.ttf", 17)

    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        text = det_labels[i].upper()+" {0:.2f}%".format(100.0*det_scores[i])
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l+1.0 for l in box_location], outline=label_color_map[det_labels[i]])

        text_size = font.getsize(text)
        text_location = [box_location[0]+2.0, box_location[1]-text_size[1]]
        textbox_location = [box_location[0], box_location[1]-text_size[1], box_location[0]+text_size[0]+4.0, box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=text, fill='white', font=font)

    del draw
    return annotated_image