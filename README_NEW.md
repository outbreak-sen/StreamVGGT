# 环境配置
StreamVGGT

~~~
git clone https://github.com/wzzheng/StreamVGGT.git
cd StreamVGGT
conda create -n StreamVGGT python=3.11 cmake=3.14.0
conda activate StreamVGGT 
pip install -r requirements.txt
conda install 'llvm-openmp<16'
mkdir ckpt
下载vggt和streamvggt权重
├── ckpt/
|   ├── model.pt
|   └── checkpoints.pth
~~~

# 运行
100帧会outofmemory
~~~
python demo_traj.py --image_folder /share/datasets/TUM/Dynamics/rgbd_dataset_freiburg2_desk_with_person/rgb --max_frames 50 --sample_interval 10 --tum
python demo_traj.py --image_folder /share/datasets/TUM/Dynamics/rgbd_dataset_freiburg2_desk_with_person/rgb --max_frames 50 --sample_interval 10 
~~~
