# python demo/image_demo.py \
#     tests/data/coco/000000000785.jpg \
#     td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
#     td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
#     --out-file vis_results.jpg \
#     --draw-heatmap

# python demo/topdown_demo_with_mmdet.py \
#     demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
#     https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
#     configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
#     --input tests/data/coco/000000196141.jpg \
#     --output-root vis_results/ --show

# python demo/image_demo.py \
#     tests/data/coco/000000000785.jpg \
#     configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
#     Demo_model/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth \
#     --out-file vis_results.jpg

python demo/inferencer_demo.py /data/muxiangyu/TryonDatasets/datasets/FashionDataset_frames_crop \
    --pose2d configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
    --pose2d-weights Demo_model/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth --det-model yolox_l_8x8_300e_coco  --pred-out-dir vis_results/crowdpose --vis-out-dir vis_results/crowdpose