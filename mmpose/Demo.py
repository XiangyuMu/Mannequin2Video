from mmpose.apis import MMPoseInferencer
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
# register_all_modules()
img_path = 'tests/data/coco/000000000785.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose2d="configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py",
                              pose2d_weights="Demo_model/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth",
                              det_model="yolox_l_8x8_300e_coco",
                              det_cat_ids=[0]
                              )

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)