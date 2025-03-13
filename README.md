![x2](https://github.com/user-attachments/assets/e705608a-9ae8-4527-a383-f49f218ebaa2)

# 🔒 Inversion Attack - Tấn Công Đảo Ngược

## 📌 Giới Thiệu  
**Inversion Attack** là một kỹ thuật tấn công trong lĩnh vực bảo mật và học máy, nhằm tái tạo dữ liệu gốc từ đầu ra hoặc trọng số của một mô hình AI.

![sensors-22-07157-g005](https://github.com/user-attachments/assets/045e2569-0967-4b77-82c4-58116555e1ad)

**Sử dụng 2 loại tấn công 🔍 Deep Gradient Leakage (DGL) & Improved Deep Gradient Leakage (iDGL)**

### 1. Deep Gradient Leakage (DGL)  
**DGL** là một phương pháp tấn công đảo ngược dữ liệu từ gradient của mô hình.  
- Dựa trên việc khai thác gradient trong quá trình huấn luyện.  
- Cho phép kẻ tấn công tái dựng hình ảnh hoặc dữ liệu gốc từ thông tin gradient.  
- Được đề xuất trong bài báo *"Deep Leakage from Gradients" (Zhu et al., 2019)*.  

🔹 **Cơ chế hoạt động:**  
1. Kẻ tấn công có thể truy cập gradient của mô hình.  
2. Sử dụng một quá trình tối ưu hóa để khôi phục dữ liệu gốc.  
3. DGL có thể tái tạo hình ảnh với độ chính xác cao, ngay cả trên dữ liệu phức tạp.  

🔹 **Hạn chế:**  
- Độ nhạy cao với batch size (batch nhỏ dễ bị tấn công hơn).  
- Không hoạt động tốt khi có nhiễu hoặc kỹ thuật phòng thủ mạnh.  

---

### 2. Improved Deep Gradient Leakage (iDGL)  
**iDGL** là phiên bản nâng cấp của DGL, giúp tăng cường khả năng tấn công bằng cách tối ưu hóa tốt hơn.  
- Được phát triển để khai thác gradient một cách hiệu quả hơn.  
- Sử dụng thuật toán tối ưu hóa nâng cao giúp phục hồi dữ liệu chính xác hơn.  
- Có thể vượt qua một số phương pháp phòng thủ cơ bản như thêm nhiễu hoặc làm mờ gradient.  

🔹 **Cơ chế hoạt động:**  
1. Tận dụng kỹ thuật tối ưu hóa gradient tốt hơn.  
2. Cải thiện khả năng tái tạo chi tiết ảnh so với DGL.  
3. Ít bị ảnh hưởng bởi batch size hoặc một số kỹ thuật phòng thủ yếu.  

🔹 **So sánh DGL và iDGL:**  
![DLG_iDLG](https://github.com/user-attachments/assets/3d411082-415a-4455-ae29-d73cae7ef362)

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
