**Input:**
*   Một ảnh chưa gán nhãn `x_u`.
*   Mô hình: Student `S` và 2 Teacher `T1`, `T2`. Cả 3 cùng kiến trúc (Encoder-Decoder).

**Ý tưởng chính:** Đưa mô hình Student tạo ra một feature map giống feature map được tổng hợp từ 2 mô hình Teacher qua consistency regularization. Khác với cách pseudo-label của paper gốc.

---

#### **Phần 1: Tạo cái Target Feature Map từ 2 Teacher**
Mục tiêu: Tạo 'ground truth' tốt cho Student

1.  **Augment ảnh nhẹ:**
    *   Lấy ảnh gốc `x_u`, dùng **Weak Augmentation** (lật, crop nhẹ...).
    *   -> Ra được `x_u_weak`.
    
2.  **Cho chạy qua 2 Teacher:**
    *   Đưa `x_u_weak` vào cả `T1` và `T2` song song.
    *   Flow: `Input -> Encoder -> Decoder`.
    *   **Quan trọng:** Dừng ngay trước lớp classification cuối cùng. Cần dùng **feature map** từ output của Decoder.
    *   -> Ra được 2 cái feature map `f_t1` và `f_t2`. (Kích thước ví dụ: 64x64x256).

3.  **Chuẩn hóa L2 (bắt buộc):**
    *   Với mỗi pixel, lấy vector feature 256 chiều của nó rồi chuẩn hóa L2 cho nó có độ dài = 1.
    *   -> Ra được `f_hat_t1` và `f_hat_t2`.
    *   Note: Bước này để đưa mọi thứ lên hypersphere, chuẩn bị cho Slerp.

4.  **Kết hợp 2 feature map bằng SLERP:**
    *   Tại mỗi pixel, lấy 2 vector feature tương ứng từ `f_hat_t1` và `f_hat_t2` rồi interpolate ra vector giữa bằng Slerp.
    *   Ý tưởng công thức Slerp là tìm đường đi "ngắn nhất" trên mặt cầu giữa 2 vector.
    *   -> Ra được **`f_target`** là target cuối cùng
    *   **Note:** Hệ số trộn `alpha` có thể không để 0.5 cố định mà linh động được. 1 cách là tính từ confidence của từng teacher tại pixel đó. Mô hình tự tin hơn thì `alpha` cao hơn ở đó.x   

---

#### **Phần 2: Student**
Mục tiêu: Tạo ra một cái feature map để so sánh với target ở trên

1.  **Augment ảnh:**
    *   Lấy ảnh gốc `x_u`, nhưng dùng **Strong Augmentation** (RandAugment, đổi màu, Cutout...).
    *   -> Ra được `x_u_strong`.

2.  **Cho chạy qua Student:**
    *   Đưa `x_u_strong` vào mạng Student `S`.
    *   Tương tự: `Input -> Encoder -> Decoder -> Lấy feature map f_s`.
    *   Chuẩn hóa L2 -> `f_hat_s`.

---

#### **Phần 3: Tính Loss**
So sánh: `f_target` và `f_hat_s`

1.  **Tính Feature Consistency Loss:**
    *   Tính **MSE (Mean Squared Error)** giữa cái target `f_target` và dự đoán `f_hat_s`.
    *   `L_consistency = MSE(f_hat_s, f_target)`
    *   Mục tiêu: Kéo `f_hat_s` về gần với `f_target`.

2.  **Loss tổng:**
    *   Cộng `L_consistency` với `L_supervised` (tính trên batch ảnh có nhãn).
    *   `L_total = L_supervised + λ * L_consistency`

---

#### **Phần 4: Cập nhật Weight**

1.  **Cập nhật Student:**
    *   Backprop từ `L_total` để update weight cho Student.

2.  **Cập nhật Teacher:**
    *   **Không backprop Teacher**
    *   Dùng **EMA (Exponential Moving Average)**. Lấy weight mới của Student để update từ từ cho Teacher.
    *   `θ_teacher_new = β * θ_teacher_old + (1 - β) * θ_student_new`
