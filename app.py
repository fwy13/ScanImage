from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Thư mục để lưu trữ ảnh tải lên
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route cho trang chủ
@app.route('/')
def home():
    return render_template('index.html', result_image=None)

# Route để xử lý ảnh tải lên
@app.route('/upload', methods=['POST'])
def upload_image():
    file2 = request.files['file2']

    # Lưu ảnh tải lên vào thư mục uploads
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'b.jpg')
    file2.save(path2)

    # Xử lý ảnh
    result_image_path = process_images(path2)

    # Trả về kết quả trên trang index.html
    return render_template('index.html', result_image=result_image_path)

# Hàm xử lý ảnh
def process_images(path2):
    # Đọc ảnh mẫu (a.jpg) và ảnh lớn (b.jpg)
    img1 = cv2.imread("./template.jpg", 0)  # Ảnh mẫu (chuyển sang ảnh xám)
    img2 = cv2.imread(path2, 0)  # Ảnh lớn (chuyển sang ảnh xám)

    # Kiểm tra xem ảnh có được đọc thành công không
    if img1 is None or img2 is None:
        return None

    # Giảm kích thước ảnh (ví dụ: 50%)
    scale_percent = 50
    img1 = resize_image(img1, scale_percent)
    img2 = resize_image(img2, scale_percent)

    # Khởi tạo SIFT detector
    sift = cv2.SIFT_create()

    # Tìm keypoints và descriptors với SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Kiểm tra xem có tìm thấy keypoints và descriptors không
    if des1 is None or des2 is None:
        return None

    # Sử dụng BFMatcher để so khớp các descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Áp dụng tỷ lệ Lowe's ratio test để lọc các matches tốt
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Kiểm tra số lượng matches tốt
    if len(good_matches) > 10:
        # Lấy tọa độ của các keypoints tốt nhất
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Tìm ma trận homography để xác định vùng khớp
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Lấy kích thước của ảnh mẫu
        h, w = img1.shape

        # Tạo các điểm góc của ảnh mẫu
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Áp dụng phép biến đổi homography để tìm vùng khớp trong ảnh lớn
        dst = cv2.perspectiveTransform(pts, M)

        # Vẽ hình chữ nhật màu đỏ xung quanh vùng khớp
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

        # Lưu ảnh kết quả
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(result_image_path, img2_color)

        return 'result.jpg'
    else:
        return None

# Hàm giảm kích thước ảnh
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Route để phục vụ ảnh kết quả
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')