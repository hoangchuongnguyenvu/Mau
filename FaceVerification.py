# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import uuid

# Đường dẫn tới models
MODELS_DIR = "models"
# Đường dẫn tới config
CONFIG_DIR = "config"

# Load Firebase credentials
cred = credentials.Certificate(os.path.join(CONFIG_DIR, "firebase_credentials.json"))

# Thiết lập trang
st.set_page_config(layout="wide", page_title="Hệ thống Quản lý Sinh viên")

# CSS chung
st.markdown("""
<style>
    .table-container {
        display: flex;
        justify-content: center;
        width: 100%;
        overflow-x: auto;
    }
    .dataframe {
        font-size: 14px;
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe th, .dataframe td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    .dataframe td:nth-child(3), .dataframe td:nth-child(4) {
        text-align: center;
    }
    .dataframe img {
        max-width: 80px;
        max-height: 80px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    .search-result {
        margin: 20px 0;
    }
    .result-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 2px solid #333;
        padding-bottom: 5px;
    }
    .result-table {
        display: table;
        width: 100%;
        border-collapse: collapse;
    }
    .result-row {
        display: table-row;
    }
    .result-cell {
        display: table-cell;
        padding: 10px;
        vertical-align: middle;
        border-right: 1px solid #ddd;
    }
    .result-cell:last-child {
        border-right: none;
    }
    .info-label {
        font-weight: bold;
        margin-right: 10px;
    }
    .image-container {
        text-align: center;
    }
    .image-label {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .image-container img {
        max-width: 150px;
        max-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo Firebase (chỉ thực hiện một lần)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {'storageBucket': 'hchuong.appspot.com'})

# Kết nối đến Firestore và Storage
db = firestore.client()
bucket = storage.bucket()

# Khởi tạo session state
if 'current_action' not in st.session_state:
    st.session_state.current_action = None

# Helper Functions
def normalize_text(text):
    if not text:
        return ""
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
    }
    text = text.lower()
    for k, v in vietnamese_map.items():
        text = text.replace(k, v)
    return text

def upload_image(file):
    if file is not None:
        file_name = str(uuid.uuid4()) + "." + file.name.split(".")[-1]
        blob = bucket.blob(file_name)
        blob.upload_from_file(file)
        blob.make_public()
        return blob.public_url
    return None

def init_haar_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

def init_yunet_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        raise FileNotFoundError("YuNet or SFace model file not found")

    face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (0, 0), 0.6, 0.3, 1)
    face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    
    return face_detector, face_recognizer

def init_sface():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sface_path = os.path.join(current_dir, "face_recognition_sface_2021dec.onnx")
    if not os.path.exists(sface_path):
        raise FileNotFoundError("SFace model file not found")
    return cv2.FaceRecognizerSF.create(sface_path, "")

def get_student_data():
    students_ref = db.collection("Students")
    students = students_ref.get()
    table_data = []
    for student in students:
        student_data = student.to_dict()
        table_data.append({
            "ID": student.id,
            "Name": student_data.get("Name", ""),
            "TheSV": student_data.get("TheSV", ""),
            "ChanDung": student_data.get("ChanDung", "")
        })
    return table_data

# Các hàm xử lý khuôn mặt
def detect_face_haar(image, cascade):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces

def detect_recognize_face_yunet(image, face_detector, face_recognizer):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    
    if faces is not None and len(faces) > 0:
        face = faces[0]
        aligned_face = face_recognizer.alignCrop(img_rgb, face)
        feature = face_recognizer.feature(aligned_face)
        return img_rgb, faces[0], feature
    return img_rgb, None, None

def compare_faces(feature1, feature2, face_recognizer):
    cosine_score = face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    return cosine_score

def draw_faces(img, faces, is_haar=True):
    if is_haar:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        if faces is not None:
            bbox = faces[:4].astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    return img

# Tạo menu chính
st.sidebar.title("Menu Chính")
menu_options = [
    "1. Quản lý Sinh viên",
    "2. Xác thực Khuôn mặt",
    "3. Nhận diện Sinh viên trong Lớp"
]
selected_menu = st.sidebar.radio("Chọn chức năng:", menu_options)

# Xử lý hiển thị theo menu được chọn
if selected_menu == "1. Quản lý Sinh viên":
    st.header("1. Quản lý Sinh viên")
    
    # Tạo các nút cho các chức năng
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Thêm Sinh viên"):
            st.session_state.current_action = 'add'
    with col2:
        if st.button("Tìm kiếm Sinh viên"):
            st.session_state.current_action = 'search'

    # Chức năng thêm sinh viên mới
    if st.session_state.current_action == 'add':
        st.subheader("Thêm Sinh viên mới")
        
        st.info("""
        📝 **Hướng dẫn thêm sinh viên:**
        - ID: Nhập mã số sinh viên (không được trùng với ID đã có)
        - Tên: Nhập đầy đủ họ và tên sinh viên
        - Thẻ Sinh viên & Ảnh Chân dung: Upload file ảnh (JPG, PNG, JPEG)
        
        ⚠️ **Lưu ý:** 
        - Tất cả các trường thông tin đều bắt buộc
        - Kích thước file ảnh tối đa 200MB
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.text_input("ID")
        with col2:
            new_name = st.text_input("Tên")
        
        new_thesv = st.file_uploader("Thẻ Sinh viên", type=["jpg", "png", "jpeg"])
        new_chandung = st.file_uploader("Ảnh Chân dung", type=["jpg", "png", "jpeg"])

        if st.button("Xác nhận thêm"):
            if new_id and new_name and new_thesv and new_chandung:
                doc_ref = db.collection("Students").document(new_id).get()
                if doc_ref.exists:
                    st.error(f"ID {new_id} đã tồn tại! Vui lòng chọn ID khác.")
                else:
                    thesv_url = upload_image(new_thesv)
                    chandung_url = upload_image(new_chandung)
                    db.collection("Students").document(new_id).set({
                        "Name": new_name,
                        "TheSV": thesv_url,
                        "ChanDung": chandung_url
                    })
                    st.success("Đã thêm sinh viên mới!")
                    st.session_state.current_action = None
                    st.rerun()
            else:
                st.warning("Vui lòng điền đầy đủ thông tin!")

    # Chức năng tìm kiếm
    elif st.session_state.current_action == 'search':
        st.subheader("Tìm kiếm Sinh viên")
        
        st.info("""
        🔍 **Hướng dẫn tìm kiếm:**
        1. Tìm theo ID: 
           - Nhập chính xác mã số sinh viên
        
        2. Tìm theo Tên:
           - Tìm theo họ: Nhập trực tiếp (vd: Ho, Hoang)
           - Tìm theo tên: Thêm # trước tên (vd: #Chuong)
        
        3. Tìm kết hợp:
           - Có thể nhập cả ID và tên để tìm chính xác hơn
        
        ⚠️ **Lưu ý:** 
        - Không phân biệt chữ hoa/thường
        - Không phân biệt dấu tiếng Việt
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            search_id = st.text_input("Nhập ID sinh viên")
        with col2:
            search_name = st.text_input("Nhập tên sinh viên (thêm # ở đầu để tìm theo tên)")

        if st.button("Xác nhận tìm kiếm"):
            found_students = []
            
            if search_id:
                student = db.collection("Students").document(search_id).get()
                if student.exists:
                    student_data = student.to_dict()
                    if search_name:
                        search_text = search_name.strip()
                        is_search_by_last_name = not search_text.startswith('#')
                        
                        if search_text.startswith('#'):
                            search_text = search_text[1:].strip()
                        
                        normalized_search = normalize_text(search_text)
                        full_name = student_data.get('Name', '')
                        name_parts = full_name.split()
                        
                        if is_search_by_last_name:
                            if name_parts and normalized_search in normalize_text(name_parts[0]):
                                found_students.append((search_id, student_data))
                        else:
                            if name_parts and normalize_text(name_parts[-1]) == normalized_search:
                                found_students.append((search_id, student_data))
                    else:
                        found_students.append((search_id, student_data))
            
            elif search_name:
                search_text = search_name.strip()
                is_search_by_last_name = not search_text.startswith('#')
                
                if search_text.startswith('#'):
                    search_text = search_text[1:].strip()
                    
                normalized_search = normalize_text(search_text)
                all_students = db.collection("Students").stream()
                
                for student in all_students:
                    student_data = student.to_dict()
                    full_name = student_data.get('Name', '')
                    name_parts = full_name.split()
                    
                    if is_search_by_last_name:
                        if name_parts and normalize_text(name_parts[0]).startswith(normalized_search):
                            found_students.append((student.id, student_data))
                    else:
                        if name_parts and normalize_text(name_parts[-1]) == normalized_search:
                            found_students.append((student.id, student_data))

            if found_students:
                for student_id, student_data in found_students:
                    st.markdown(f"""
                    <div class="search-result">
                        <div class="result-header">Thông tin sinh viên</div>
                        <div class="result-table">
                            <div class="result-row">
                                <div class="result-cell">
                                    <span class="info-label">ID:</span>
                                    <span>{student_id}</span>
                                </div>
                                <div class="result-cell">
                                    <span class="info-label">Tên:</span>
                                    <span>{student_data.get('Name', '')}</span>
                                </div>
                                <div class="result-cell">
                                    <div class="image-container">
                                        <div class="image-label">Thẻ Sinh viên</div>
                                        <img src="{student_data.get('TheSV', '')}" alt="Thẻ Sinh viên">
                                    </div>
                                </div>
                                <div class="result-cell">
                                    <div class="image-container">
                                        <div class="image-label">Ảnh Chân dung</div>
                                        <img src="{student_data.get('ChanDung', '')}" alt="Ảnh Chân dung">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy sinh viên phù hợp!")

    # Hiển thị bảng dữ liệu
    st.subheader("Danh sách Sinh viên")
    table_data = get_student_data()
    df = pd.DataFrame(table_data)

    df['Edit'] = False
    df['Delete'] = False

    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Edit": st.column_config.CheckboxColumn("Chỉnh sửa", default=False, width="small"),
            "Delete": st.column_config.CheckboxColumn("Xóa", default=False, width="small"),
            "TheSV": st.column_config.ImageColumn("Thẻ SV", help="Thẻ sinh viên", width="medium"),
            "ChanDung": st.column_config.ImageColumn("Chân dung", help="Ảnh chân dung", width="medium"),
            "ID": st.column_config.TextColumn("ID", help="ID sinh viên", width="medium"),
            "Name": st.column_config.TextColumn("Tên", help="Tên sinh viên", width="large"),
        },
        disabled=["ID", "Name", "TheSV", "ChanDung"],
        use_container_width=True,
        num_rows="dynamic"
    )

    # Xử lý chỉnh sửa và xóa
    students_to_edit = edited_df[edited_df['Edit']]
    if not students_to_edit.empty:
        for _, student in students_to_edit.iterrows():
            st.subheader(f"Chỉnh sửa thông tin cho sinh viên: {student['Name']}")
            
            st.info("""
            ✏️ **Hướng dẫn chỉnh sửa:**
            - ID: Có thể thay đổi (không được trùng với ID khác)
            - Tên: Nhập tên mới cần thay đổi
            - Ảnh: Chỉ cần upload khi muốn thay đổi ảnh mới
            
            ⚠️ **Lưu ý:**
            - Nếu không upload ảnh mới, ảnh cũ sẽ được giữ nguyên
            - Sau khi thay đổi ID, sinh viên sẽ được cập nhật với ID mới
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                edit_id = st.text_input(f"ID mới cho {student['ID']}", value=student['ID'])
            with col2:
                edit_name = st.text_input(f"Tên mới cho {student['ID']}", value=student['Name'])
            
            edit_thesv = st.file_uploader(f"Thẻ Sinh viên mới cho {student['ID']}", type=["jpg", "png", "jpeg"])
            edit_chandung = st.file_uploader(f"Ảnh Chân dung mới cho {student['ID']}", type=["jpg", "png", "jpeg"])

            if st.button(f"Cập nhật cho {student['ID']}"):
                update_data = {"Name": edit_name}
                if edit_thesv:
                    thesv_url = upload_image(edit_thesv)
                    update_data["TheSV"] = thesv_url
                if edit_chandung:
                    chandung_url = upload_image(edit_chandung)
                    update_data["ChanDung"] = chandung_url
                
                if edit_id != student['ID']:
                    current_data = db.collection("Students").document(student['ID']).get().to_dict()
                    current_data.update(update_data)
                    db.collection("Students").document(edit_id).set(current_data)
                    db.collection("Students").document(student['ID']).delete()
                    st.success(f"Đã cập nhật thông tin và ID sinh viên từ {student['ID']} thành {edit_id}!")
                else:
                    db.collection("Students").document(student['ID']).update(update_data)
                    st.success(f"Đã cập nhật thông tin sinh viên {student['ID']}!")
                
                st.rerun()

    students_to_delete = edited_df[edited_df['Delete']]
    if not students_to_delete.empty:
        for _, student in students_to_delete.iterrows():
            st.subheader(f"Xác nhận xóa sinh viên: {student['Name']}")
            
            st.warning("""
            ⚠️ **Cảnh báo:**
            - Thao tác xóa không thể hoàn tác
            - Tất cả thông tin của sinh viên sẽ bị xóa vĩnh viễn
            - Vui lòng kiểm tra kỹ trước khi xác nhận xóa
            """)
            
            if st.button(f"Xác nhận xóa {student['ID']}"):
                db.collection("Students").document(student['ID']).delete()
                st.success(f"Đã xóa sinh viên {student['ID']}!")
                st.rerun()

elif selected_menu == "2. Xác thực Khuôn mặt":
    st.title("Ứng dụng So sánh Ảnh Chân dung và Thẻ Sinh viên")

    haar_cascade = init_haar_cascade()
    yunet_detector, sface_recognizer = init_yunet_sface()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Ảnh Chân dung")
        portrait_image = st.file_uploader("Tải lên ảnh chân dung", type=['jpg', 'jpeg', 'png'])

    with col2:
        st.header("Ảnh Thẻ Sinh viên")
        id_image = st.file_uploader("Tải lên ảnh thẻ sinh viên", type=['jpg', 'jpeg', 'png'])

    check_button = st.button("Kiểm tra")

    if portrait_image and id_image and check_button:
        portrait_img, portrait_faces = detect_face_haar(portrait_image, haar_cascade)
        id_img, id_face, id_feature = detect_recognize_face_yunet(id_image, yunet_detector, sface_recognizer)

        if len(portrait_faces) > 0 and id_face is not None:
            largest_face = max(portrait_faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            portrait_face_img = portrait_img[y:y+h, x:x+w]
            portrait_face_feature = sface_recognizer.feature(cv2.resize(portrait_face_img, (112, 112)))
            
            similarity_score = compare_faces(portrait_face_feature, id_feature, sface_recognizer)

            st.header("Kết quả So sánh")
            st.write(f"Độ tương đồng: {similarity_score:.4f}")

            if similarity_score > 0.3:
                st.success("Ảnh chân dung và ảnh thẻ sinh viên KHỚP!")
                color = (0, 255, 0)
            else:
                st.error("Ảnh chân dung và ảnh thẻ sinh viên KHÔNG KHỚP!")
                color = (0, 0, 255)

            portrait_img_with_rect = draw_faces(portrait_img.copy(), [largest_face])
            id_img_with_rect = draw_faces(id_img.copy(), id_face, is_haar=False)

            col1, col2 = st.columns(2)
            with col1:
                st.image(portrait_img_with_rect, caption="Ảnh Chân dung", use_column_width=True)
            with col2:
                st.image(id_img_with_rect, caption="Ảnh Thẻ Sinh viên", use_column_width=True)
        else:
            st.error("Không thể phát hiện khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.")
    elif check_button:
        st.warning("Vui lòng tải lên cả ảnh chân dung và ảnh thẻ sinh viên trước khi kiểm tra.")

else:  # "3. Nhận diện Sinh viên trong Lớp"
    st.title("Tìm kiếm Sinh viên trong Ảnh Lớp học")

    def process_student_image(image_data, cascade, face_recognizer):
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_img = img_rgb[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (112, 112))
            feature = face_recognizer.feature(face_img)
            return img_rgb, (x, y, w, h), feature
        return img_rgb, None, None

    def process_class_image(image, cascade, face_recognizer):
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_features = []
        if len(faces) > 0:
            for face in faces:
                x, y, w, h = face
                face_img = img_rgb[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (112, 112))
                feature = face_recognizer.feature(face_img)
                face_features.append((face, feature))
        
        return img_rgb, faces, face_features

    def crop_face(img, face):
        x, y, w, h = face
        padding = int(min(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        return img[int(y1):int(y2), int(x1):int(x2)]

    def draw_results(img, faces, matches):
        img_copy = img.copy()
        for face, matched_students in zip(faces, matches):
            x, y, w, h = face
            
            if matched_students:
                best_match = max(matched_students, key=lambda x: x[1])
                student_name, score = best_match
                
                color = (0, 255, 0)
                text = f"{student_name} ({score:.2f})"
            else:
                color = (0, 0, 255)
                text = "Unknown"
            
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x
            text_y = y - 10 if y - 10 > text_size[1] else y + h + 20
            
            cv2.rectangle(img_copy, 
                         (text_x, text_y - text_size[1] - 4),
                         (text_x + text_size[0], text_y + 4),
                         color, -1)
            cv2.putText(img_copy, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_copy

    # Khởi tạo các mô hình
    haar_cascade = init_haar_cascade()
    sface_recognizer = init_sface()

    # Giao diện người dùng
    st.header("Tải lên Ảnh Lớp học")
    class_image = st.file_uploader("Chọn ảnh lớp học", type=['jpg', 'jpeg', 'png'])

    threshold = st.slider("Ngưỡng nhận dạng (0-1)", 0.0, 1.0, 0.3, 0.001)
    search_button = st.button("Tìm kiếm")

    if class_image and search_button:
        try:
            students_ref = db.collection("Students")
            students = students_ref.get()
            
            class_img, class_faces, class_features = process_class_image(
                class_image, haar_cascade, sface_recognizer)
            
            if len(class_faces) > 0:
                face_matches = [[] for _ in class_faces]
                
                with st.spinner('Đang xử lý tất cả sinh viên...'):
                    for student in students:
                        student_data = student.to_dict()
                        student_name = student_data.get('Name')
                        chandung_url = student_data.get('ChanDung')
                        
                        if chandung_url:
                            response = requests.get(chandung_url)
                            if response.status_code == 200:
                                _, _, student_feature = process_student_image(
                                    response.content, haar_cascade, sface_recognizer)
                                
                                if student_feature is not None:
                                    for i, (_, class_feature) in enumerate(class_features):
                                        score = compare_faces(student_feature, class_feature, sface_recognizer)
                                        if score > threshold:
                                            face_matches[i].append((student_name, score))
                
                result_img = draw_results(class_img, class_faces, face_matches)
                
                st.header("Kết quả Nhận diện")
                st.image(result_img, caption="Kết quả nhận diện trong lớp học", use_column_width=True)
                
                matched_faces = sum(1 for matches in face_matches if matches)
                st.write(f"Đã nhận diện được {matched_faces} khuôn mặt trong {len(class_faces)} khuôn mặt phát hiện được")
                
                if matched_faces > 0:
                    st.header("Các khuôn mặt được nhận dạng:")
                    
                    cols = st.columns(4)
                    col_idx = 0
                    
                    for i, (face, matches) in enumerate(zip(class_faces, face_matches)):
                        if matches:
                            best_match = max(matches, key=lambda x: x[1])
                            student_name, score = best_match
                            
                            face_img = crop_face(class_img, face)
                            
                            with cols[col_idx]:
                                st.image(face_img, caption=f"{student_name}\n({score:.2f})")
                                col_idx = (col_idx + 1) % 4
                                
                                if col_idx == 0:
                                    cols = st.columns(4)
                                    
            else:
                st.error("Không thể phát hiện khuôn mặt trong ảnh lớp học")
                
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {str(e)}")
    elif search_button:
        st.warning("Vui lòng tải lên ảnh lớp học trước khi tìm kiếm")