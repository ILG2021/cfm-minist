import os
import tarfile
import urllib.request

# 定义下载 URL 和目标路径
url = "http://www.di.ens.fr/~lelarge/MNIST.tar.gz"
download_path = "./MNIST.tar.gz"
extract_path = "./data"

# 创建目标目录
os.makedirs(extract_path, exist_ok=True)

# 下载文件
print("Downloading MNIST dataset...")
urllib.request.urlretrieve(url, download_path)
print("Download completed.")

# 解压文件
print("Extracting MNIST dataset...")
with tarfile.open(download_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
print("Extraction completed.")

# 删除压缩包（可选）
os.remove(download_path)
print("Temporary tar.gz file removed.")

# 完成
print(f"MNIST dataset is ready at: {extract_path}")
