#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import cv2  # 引入 OpenCV 解决图片格式兼容性问题
# --- 页面基本配置 ---
st.set_page_config(page_title="人脸识别系统", page_icon="🧐", layout="wide")
st.title("🧐 计算机视觉：人脸检测与特征提取")
st.write("上传一张图片，系统将自动检测人脸位置、绘制边框并提取面部特征向量。")
# --- 核心功能函数 ---
def process_image(uploaded_file):
    """
    处理上传的图片：检测人脸 -> 绘制边框 -> 提取特征
    使用 OpenCV 读取图片以解决 'Unsupported image type' 错误
    """
    try:
        # ==========================================
        # 关键修复：使用 OpenCV 处理字节流
        # ==========================================
        # 1. 将上传的文件流读取为字节数组
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # 2. 使用 OpenCV 解码图片
        # cv2.IMREAD_COLOR 表示强制转换为彩色图片 (BGR格式)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # 3. 检查图片是否读取成功
        if img_array is None:
            st.error("无法读取图片，请确认文件未损坏。")
            return None, None
        # 4. 颜色空间转换
        # OpenCV 默认读取为 BGR，face_recognition 需要 RGB
        image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # ==========================================
        # 修复结束
        # ==========================================
        # 5. 人脸检测
        # model='hog' 速度快，CPU友好；model='cnn' 精度高但需GPU
        face_locations = face_recognition.face_locations(image, model='hog')
        # 6. 人脸特征提取
        # 返回的是 128 维的特征向量列表
        face_encodings = face_recognition.face_encodings(image, face_locations)
        # 7. 准备绘图
        # 将处理好的 RGB 数组转换为 PIL 图片进行绘制
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        results = []
        # 遍历检测到的每一张脸
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 绘制矩形框 (边框颜色红色，线宽2)
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
            # 记录结果
            results.append({
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "encoding": face_encoding
            })
        return pil_image, results
    except Exception as e:
        import traceback
        st.error(f"处理图片时发生未知错误: {e}")
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None, None
# --- 前端交互界面 ---
# 侧边栏说明
with st.sidebar:
    st.header("📋 使用说明")
    st.info(
        """
        1. 点击下方按钮上传图片。
        2. 支持 JPG/PNG 格式。
        3. 等待几秒后查看检测结果。
        """
    )
    st.caption("Powered by dlib & face_recognition")
# 文件上传控件
uploaded_file = st.file_uploader("请选择一张包含人脸的图片...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # 展示原始图片（预览）
    st.subheader("1. 原始图片")
    st.image(uploaded_file, caption="上传的原始图片", use_column_width=True)
    # 开始处理按钮
    if st.button("开始检测与特征提取", type="primary"):
        with st.spinner('正在分析中，请稍候...'):
            # 调用核心处理函数
            result_image, face_data = process_image(uploaded_file)
            if result_image is not None:
                st.success(f"处理完成！共检测到 {len(face_data)} 张人脸。")
                # 使用列布局展示结果
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("2. 检测结果")
                    st.image(result_image, caption="人脸定位结果", use_column_width=True)
                with col2:
                    st.subheader("3. 特征提取详情")
                    if len(face_data) > 0:
                        for i, face in enumerate(face_data):
                            with st.expander(f"人脸 #{i+1} 详情", expanded=(i==0)):
                                st.write("**坐标位置:**", face['location'])
                                st.write("**特征向量 (前10维预览):**")
                                encoding_preview = face['encoding'][:10]
                                st.code(np.array2string(encoding_preview, precision=4, separator=', '))
                                st.write("**特征值分布示例:**")
                                st.bar_chart(face['encoding'][:20])
                    else:
                        st.warning("未检测到人脸，请尝试换一张更清晰的正面照片。")


# In[ ]:




