# a demo of using Conditional Flow Matching to generate mnist data

## 安装
``
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``

``
pip install torchcfm
``

## 下载数据集
``
python download_mnist.py
``

## 训练
``
python train.py
``

## 推理
``
python inference.py
``
