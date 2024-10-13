import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from scipy import ndimage


def Noising_image(image, prob):
    output = np.copy(image)
    # Xác suất để một pixel bị đổi thành muối hoặc tiêu
    salt_prob = prob / 2
    pepper_prob = prob / 2
    
    # Thêm muối (trắng)
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    output[salt_coords[0], salt_coords[1]] = 255

    # Thêm tiêu (đen)
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    output[pepper_coords[0], pepper_coords[1]] = 0

    return output

def Denoising_Smoothing_image(image, filter_type):
    if filter_type == 'Gaussian':
        filtered = cv2.GaussianBlur(image, (5, 5), 0)  # Smoothing
    elif filter_type == 'Median':
        filtered = cv2.medianBlur(image, 5)  # Smoothing, can reduce noise
    elif filter_type == 'Mean':
        filtered = cv2.blur(image, (5, 5))  # Smoothing
    elif filter_type == 'Bilateral':
        filtered = cv2.bilateralFilter(image, 9, 100, 100)  # Tăng sigma # Denoising while keeping edges
    return filtered

def Sharpening_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def Sobel_filter(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)  # Chuẩn hóa về dạng uint8
    return sobel.astype(np.uint8)  # Chuyển về kiểu uint8

def Prewitt_filter(image):
    # Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)
    return prewitt  #dtype: float64

def Canny_Edge_filter(image):
    canny = cv2.Canny(image, 100, 200)
    return canny  # Dtype: uint8

option_mapping = {
    'Denoising / Smoothing': ['Gaussian', 'Median', 'Mean', 'Bilateral'],
    'Sharpening': ['Sharpening'],
    'Edge Detection filter': ['Sobel', 'Prewitt', 'Canny Edge']
}

filter_functions = {
    'Sobel': Sobel_filter,
    'Prewitt': Prewitt_filter,
    'Canny Edge': Canny_Edge_filter
}

st.title('Image Processing')
image_up = st.file_uploader('Upload your image', type=['png', 'jpg', 'jpeg', 'bmp'])

if image_up is not None:

    img = np.asarray(bytearray(image_up.read()), dtype=np.uint8) 
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Đọc ảnh từ byte array

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img  

    st.markdown("""
    **Note:** Lua chon yeu cau cua ban:
    - **1. Denoising / Smoothing**
    - **2. Sharpening**
    - **3. Edge Detection filter: Sobel, Prewitt, Canny Edge**
    """)
    # Chon yeu cau theo markdown
    option = st.selectbox('Choose your option', 
                        ('Denoising / Smoothing',
                            'Sharpening',
                            'Edge Detection filter')
                            )
    
    if option == 'Denoising / Smoothing':
        st.markdown("""
        **Note:** Chuc năng tương ứng với các filter:
        - **1. Denoising:  'Bilateral**
        - **2. Smoothing: 'Gaussian', 'Median', 'Mean'**
        """)    
        filter = st.selectbox('Choose filter', option_mapping[option])
        st.markdown("""
        **Note**: Hướng dẫn sử dụng tham số nhiễu:
        - **Tỷ lệ thấp (0.01 đến 0.05)**
        - **Tỷ lệ trung bình (0.05 đến 0.1)**
        - **Tỷ lệ cao (0.1 đến 0.3)**
    """)
        prob = st.slider("Chọn xác suất thêm nhiễu (0.01 đến 0.3)", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
        st.write('You entered:', prob)
        if st.button("Plot image with filter {}".format(filter)):
            img_noise = Noising_image(img, prob)
            cols_plot = st.columns(3)
            denoised = Denoising_Smoothing_image(img_noise, filter)
            with cols_plot[0]:  # Đặt ảnh ở đầu tiên
                st.image(img, caption="Original", width=230)  # Điều chỉnh width để ảnh nhỏ hơn
            with cols_plot[1]:
                st.image(img_noise, caption="Image Noise of Original", width=230)
            with cols_plot[2]:
                st.image(denoised, caption="{} of Image Noise".format(filter), width=230)
                
    elif option == 'Sharpening':
        if st.button("Plot image with filter Sharpening"):
            filter = 'Sharpening'
            cols_plot = st.columns(2)
            sharpened = Sharpening_image(img)
            with cols_plot[0]:  # Đặt ảnh ở đầu tiên
                st.image(img, caption="Original", width=350)  # Điều chỉnh width để ảnh nhỏ hơn
            with cols_plot[-1]:
                st.image(sharpened, caption="Sharpened of Original", width=350)

    elif option == 'Edge Detection filter':
        filter = st.selectbox('Choose Edge Detection filter', option_mapping[option])

        if st.button(f"Plot image with {filter} filter"):
            cols_plot = st.columns(3)
            array_float64 = img_gray.astype(np.float64)
            array_float64_normalized = array_float64 / 255.0

            if filter == 'Prewitt':
                img_filter = filter_functions[filter](array_float64_normalized)
            else:
                img_filter = filter_functions[filter](img_gray)

            # Hiển thị 3 hình ảnh trong 3 cột
            with cols_plot[0]:
                fig, ax = plt.subplots()
                ax.imshow(img, cmap='gray')
                ax.set_title("Original")
                ax.axis('off')
                st.pyplot(fig)

            with cols_plot[1]:
                fig, ax = plt.subplots()
                ax.imshow(img_gray, cmap='gray')
                ax.set_title("Original To Gray")
                ax.axis('off')
                st.pyplot(fig)

            with cols_plot[2]:
                fig, ax = plt.subplots()
                ax.imshow(img_filter, cmap='gray')
                ax.set_title(f"{filter} of Gray Image")
                ax.axis('off')
                st.pyplot(fig)

        # Tách nút "Plot all case" ra thành một hành động riêng biệt
        if st.button("Plot all filters"):
            # Tạo array_float64 cho tính toán trước
            array_float64 = img_gray.astype(np.float64)
            array_float64_normalized = array_float64 / 255.0
            total_filters = len(option_mapping[option])
            
            # Tạo figure và lưới 3x3 (3 hàng, 3 cột)
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Kích thước có thể điều chỉnh

            for row in range(3):  # Duyệt qua 3 hàng (mỗi filter sẽ chiếm một hàng)
                current_filter = option_mapping[option][row]

                # Ảnh gốc (Original)
                axs[row, 0].imshow(img, cmap='gray')
                axs[row, 0].set_title(f"Original")
                axs[row, 0].axis('off')

                # Ảnh chuyển sang xám (Original To Gray)
                axs[row, 1].imshow(img_gray, cmap='gray')
                axs[row, 1].set_title(f"Original To Gray")
                axs[row, 1].axis('off')

                # Ảnh áp dụng filter lên ảnh xám (Filter of Original To Gray)
                if current_filter == 'Prewitt':
                    img_filter = filter_functions[current_filter](array_float64_normalized)
                else:
                    img_filter = filter_functions[current_filter](img_gray)

                axs[row, 2].imshow(img_filter, cmap='gray')
                axs[row, 2].set_title(f"{current_filter} of Gray")
                axs[row, 2].axis('off')

            st.pyplot(fig)  # Hiển thị figure



        


        


