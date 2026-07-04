# BraTS2023 Experiment Flow

Dưới đây là kế hoạch chuyển đổi và tái hiện các thử nghiệm từ BraTS2020 sang BraTS2023.
Mục tiêu là xác nhận các kỹ thuật (Ablation) hoạt động tốt trên BraTS2020 cũng sẽ mang lại hiệu quả tương tự trên BraTS2023, từ đó củng cố tính tổng quát của phương pháp.

```text
Legend
------
D = Dice  WT / TC / ET / Mean     (higher is better)
H = HD95  WT / TC / ET / Mean     (lower is better)


[ BASELINE BraTS2023 ]
(1251 subjects)
        |
        v
[ A. Data Split ]
        |
        +-- Exp2023_001 | Sequential Split
        |      Config: UNet32, Min-Max, Fixed Sampling, Dice Loss, Adam, No Aug
        |      (Tương đương Exp001 của 2020)
        |      D = ...
        |      H = ...
        |
        +-- Exp2023_002 | Random/Stratified Split
               Config: Giống Exp001 nhưng dùng split ngẫu nhiên chuẩn
               (Tương đương Exp002 của 2020)
               D = ...
               H = ...

        |
        v
[ B. Normalization ]
        |
        +-- Exp2023_002 | Min-Max
        |      D = ...
        |      H = ...
        |
        +-- Exp2023_003 | Z-score Raw
        |      (Tương đương Exp003 của 2020)
        |      D = ...
        |      H = ...
        |
        +-- Exp2023_004 | Z-score Clip
               (Tương đương Exp004 của 2020)
               D = ...
               H = ...

        |
        v
[ C. Augmentation ]
        |
        +-- Exp2023_004 | No Augmentation
        |      D = ...
        |      H = ...
        |
        +-- Exp2023_005 | Spatial Flip H/V
        |      (Tương đương Exp005 của 2020)
        |      D = ...
        |      H = ...
        |
        +-- Exp2023_006 | Spatial + Intensity Augmentation
               (Tương đương Exp006 của 2020)
               D = ...
               H = ...

        |
        v
[ Các nhóm tiếp theo: Sampling (Exp007-008), Loss (Exp009-012), Optimizer (Exp013-016), Architecture (Exp017-019), Fusion (Exp020-022)... sẽ bám sát y hệt quy trình của 2020 ]
```

### Kế hoạch huấn luyện (Step-by-step)
1. Chúng ta sẽ bắt đầu từ **Exp2023_001** (giống hệt Exp001 của 2020 với Min-Max, Dice loss, Adam).
2. Sau khi chạy xong, ghi nhận kết quả và tạo tiếp Exp2023_002.
