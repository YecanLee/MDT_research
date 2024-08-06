conda create -n MDT python==3.10
conda init
conda activate MDT

pip install -r requirements.txt

wget https://huggingface.co/shgao/MDT-XL2/resolve/main/mdt_xl2_v2_ckpt.pt

python generation_single_gpu.py --tf32