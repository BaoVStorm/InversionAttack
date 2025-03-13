![Inversion Attack](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F22%2F19%2F7157&psig=AOvVaw0xD4TpAuhB-KCoWuhq3Elg&ust=1741953388967000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCPiTs5uAh4wDFQAAAAAdAAAAABAI)

# 🔒 Inversion Attack - Tấn Công Đảo Ngược

## 📌 Giới Thiệu  
**Inversion Attack** là một kỹ thuật tấn công trong lĩnh vực bảo mật và học máy, nhằm tái tạo dữ liệu gốc từ đầu ra hoặc trọng số của một mô hình AI.  

## ⚙️ Cách Hoạt Động  
1. **Dựa trên đầu ra mô hình**:  
   - Kẻ tấn công sử dụng đầu ra của mô hình để ước lượng lại dữ liệu đầu vào.  
2. **Dựa trên trọng số mô hình**:  
   - Nếu có quyền truy cập vào tham số mô hình, có thể tái tạo dữ liệu huấn luyện.  
3. **Kỹ thuật phổ biến**:  
   - Gradient-based Optimization  
   - Model Inversion (MI)  
   - Generative Model Attacks  

## 🛠 Cấu Trúc Code  
Trong code này có 3 phần chính:

### 📂 1. `Download_Dataset.ipynb`  
- Tải các bộ dữ liệu gồm: **MNIST, CIFAR-100, LFW**.  

### 📂 2. `Attack_DGL_iDGL.py`  
- Code mô tả tấn công với hai mô hình:  
  - **DGL (Deep Gradient Leakage)**  
  - **iDGL (Improved Deep Gradient Leakage)**  
- Mô tả chi tiết cách thực hiện tấn công để tái dựng dữ liệu từ mô hình.  

### 📂 3. `Defend_DP_MG_MC.py`  
- Code mô tả các phương pháp phòng thủ chống lại tấn công DGL và iDGL:  
  - **Differential Privacy (DP)**: Thêm nhiễu vào dữ liệu để bảo vệ thông tin.  
  - **Masking Gradients (MG)**: Che giấu gradient để tránh bị khai thác.  
  - **Model Compression (MC)**: Giảm kích thước mô hình để hạn chế rò rỉ thông tin.  

---

## 📊 Kết Quả Thực Nghiệm  

Kết quả được lưu trong thư mục `results`:
- **`results/attack/`**: Chứa kết quả khi tiến hành tấn công.  
- **`results/defend/`**: Chứa kết quả khi áp dụng phương pháp phòng thủ.  

---

## 📝 Góp Ý & Liên Hệ  

📩 Bạn có thể gửi phản hồi bằng cách mở một [New Issue](../../issues/new?template=feedback.yml).  

### 📌 Hướng dẫn  
1. Nhấp vào **New Issue**.  
2. Điền thông tin theo mẫu.  
3. Nhấn **Submit**.  

💡 Nếu bạn có câu hỏi, hãy liên hệ qua email [your-email@example.com](mailto:tranvubao2004@gmail.com).  
