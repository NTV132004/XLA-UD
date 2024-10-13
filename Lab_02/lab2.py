import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Tạo ghi chú cho người dùng
category_color_mapping = {
    'buildings': 'Gray',
    'forest': 'BGR',
    'glacier': 'BGR',
    'mountain': 'Gray',
    'sea': 'HSV',
    'street': 'HSV'
}

def calculate_histogram(image, color_space='BGR', bins=256):
    if color_space == 'BGR':
        hist = []
        for i in range(3):  # Tính histogram cho mỗi kênh màu R, G, B
            hist.append(cv2.calcHist([image], [i], None, [bins], [0, 256]))
        hist = np.concatenate(hist)
    elif color_space == 'HSV':
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist = []
        for i in range(3):  # Tính histogram cho các kênh H, S, V
            hist.append(cv2.calcHist([hsv_image], [i], None, [bins], [0, 256]))
        hist = np.concatenate(hist)
    else:
        hist = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        equalized_image = cv2.equalizeHist(edges)

        hist_0 = cv2.calcHist([equalized_image], [0], None, [bins], [0, 256])
        hist.append(hist_0)
        hist.append(hist_0)
        hist.append(hist_0)
        hist = np.concatenate(hist)

    return cv2.normalize(hist, hist).flatten()



def compute_hist_list(dir_root_path):
  hist_tuples = []
  count = 0

  for dir_name in os.listdir(dir_root_path):
    dir_path = os.path.join(dir_root_path,dir_name)
    for file_name in os.listdir(dir_path):
      #save path
      file_path = os.path.join(dir_path,file_name)
      print(file_path)

      # read image
      image = cv2.imread(file_path)
      image_resized = cv2.resize(image, (256, 256))
      # image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

      # compute histogram
      color_image = category_color_mapping[dir_name]

      hist = calculate_histogram(image_resized, color_space=color_image, bins = 256)
      hist_tuples.append((file_path, hist))
  return hist_tuples

def compare_hist(hist_1, hist_2):
   
    similarity = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
    return similarity

# seg_path = '.\seg'

# hist_tuples = compute_hist_list(seg_path)


# Đọc hist_tuples
import pickle
with open('./Feature/Lab2_Hist/hist_tuples.pkl', 'rb') as f:
    hist_tuples = pickle.load(f)




def take_dist_and_path(input_path, color_img):
    distances = []
    K = 10

    img_input = cv2.imread(input_path)
    image_resized = cv2.resize(img_input, (256, 256))
    input_hist = calculate_histogram(image_resized, color_img, 256)

    for file_name, hist in hist_tuples:
        dist = compare_hist(input_hist, hist)  # So sánh với histogram
        distances.append((file_name, dist))

    # Sắp xếp khoảng cách theo phần tử thứ hai (giá trị khoảng cách)
    dist_sorted = sorted(distances, key=lambda x: x[1])[:K]
    
    return dist_sorted




# Bước 1: Tải ảnh lên
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
# Ghi chú cho người dùng
    st.markdown("""
    **Note:** The system uses specific color spaces for different categories:
    - **buildings**: Gray
    - **forest**: BGR
    - **glacier**: BGR
    - **mountain**: Gray
    - **sea**: HSV
    - **street**: HSV
    """)

    # Bước 2: Chọn danh mục hình ảnh
    category = st.selectbox("Select the image category", list(category_color_mapping.keys()))
    
    # Hiển thị màu sắc tương ứng với danh mục đã chọn
    st.write(f"Selected color space for {category}: **{category_color_mapping[category]}**")
    
    color_space = category_color_mapping[category]

    # Bước 3: Chỉ hiển thị ảnh và thực hiện tìm kiếm sau khi nhấn nút
    if st.button("Find Similar Images"):
        # Đọc và lưu ảnh tạm thời
        img = Image.open(uploaded_file)
        temp_image_path = "./temp_image.png"
        img.save(temp_image_path)

        # Hiển thị ảnh đã tải lên (sau khi nhấn nút)
        st.write("Uploaded Image:")
        cols_input = st.columns(5)  # Tạo layout cột để hiển thị ảnh nhỏ
        with cols_input[2]:  # Đặt ảnh ở giữa
            st.image(img, caption="Input Image", width=120)  # Điều chỉnh width để ảnh nhỏ hơn

        # Tính toán và hiển thị các ảnh gần nhất
        st.write("Finding images...")
        similar_images = take_dist_and_path(temp_image_path, color_space)

        st.write("Top 10 similar images:")
        cols = st.columns(5)  # Hiển thị 5 ảnh mỗi dòng

        for idx, (file_path, dist) in enumerate(similar_images):
            img_similar = Image.open(file_path)
            with cols[idx % 5]:
                st.image(img_similar, caption=f"Dist: {dist:.4f}", use_column_width=True)