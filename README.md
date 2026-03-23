# hw03
人脸检测/识别

先确认你的 Python 版本和系统方法
在 Jupyter 里用 Python 代码查看
在单元格里输入并运行：
import sys
print(sys.version)

我的Python 版本是 3.11.7（64 位，Anaconda 环境），若运行出现问题，直接问ai，非常便捷。

1. 下载匹配版本的 .whl
文件需要下载文件名包含 cp311 和 win_amd64 的 triton 包
2. 执行安装命令
把下载好的文件放到桌面，在 PowerShell 中执行：
cd C:\Users\Desktop
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl

环境配置：# 1. 安装核心依赖（管理员身份打开 PowerShell）
pip install face_recognition streamlit pillow numpy opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

#2.在 PowerShell 窗口执行命令启动应用
python -m streamlit run app.py


操作流程：
运行命令后，自动打开浏览器（地址：http://localhost:8501）；
侧边栏查看实验说明，主界面点击「上传图片」选择人脸照片；
点击「开始检测」按钮，等待几秒后查看：
标注人脸位置的图片；
每个人脸的坐标 + 128 维特征向量；
特征值分布图表。
