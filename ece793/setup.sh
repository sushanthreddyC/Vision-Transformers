export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export PATH=~/.local/bin${PATH:+:${PATH}}
sudo apt update

sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo apt update 

sudo apt install python3.11 -y

sudo apt install python3-pip -y

python3.11 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

python3.11 -m pip install -r ./requirements.txt

# Custom fixes
python3.11 -m pip install -U Pillow
python3.11 -m pip install jupyter notebook
python3.11 -m pip uninstall -y ipython prompt_toolkit
python3.11 -m pip install ipython prompt_toolkit
python3.11 -m pip install --upgrade psutil


python3.11 -m pip install mmengine==0.10.3

python3.11 -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'


python3.11 -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

mkdir -p 3rdparty
cd 3rdparty

git clone https://github.com/pytorch/vision.git ./torchvision
cd torchvision
git checkout v0.16.2
cd ..

git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
git checkout v0.9.12
cd ..

git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
git checkout v1.2.0
python3.11 -m pip install -v .
cd ..

cd mmpretrain
python3.11 demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cuda:0
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v3.3.0
python3.11 -m pip install -v .
cd ..

cd mmdetection
python3.11 -m pip install ftfy==6.1.3 regex
mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
cd ..
python3.11 demo/image_demo.py demo/demo.jpg ./configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py --weights ./checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda:0
cd ..

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v1.2.2
python3.11 -m pip install -v .
cd ..

cd mmsegmentation
mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth
cd ..
python3.11 demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py ./checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
cd ..

git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
git checkout v1.2.0
python3.11 -m pip install -v .
cd ..

cd mmagic
# resolve an issue of diffusers==0.26.1 https://github.com/open-mmlab/mmagic/issues/2110
python3.11 -m pip install diffusers==0.24.0
# test it (wget is needed)
# to speed up cpu inference
python3.11 -m pip install accelerate==0.26.1
python3.11 demo/mmagic_inference_demo.py \
--model-name stable_diffusion \
--device cuda:0 \
--text "A panda is having dinner at KFC" \
--result-out-dir ./output/sd_res.png
cd ..

# make sure you are in the 3rdparty folder
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
git checkout v1.3.1
python3.11 -m pip install -v .
cd ..

cd mmpose
python3.11 demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py \
https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
--input tests/data/animalpose/ca110.jpeg \
--show --draw-heatmap --det-cat-id=15 \
--device cuda:0

cd ..
cd ..
cd ..
jupyter notebook
