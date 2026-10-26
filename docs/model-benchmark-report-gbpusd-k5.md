# รายงานผลการเปรียบเทียบสถาปัตยกรรมโมเดล (Phase 2 Model Benchmark): GBPUSD k=5

วันที่: 26 ต.ค. 2026 · ข้อมูล: Dukascopy M1 (ก.พ. 2024 – ต.ค. 2026 รวม 1,000,000 แท่ง) · ฟีเจอร์: Vol Squeeze Set (27 ฟีเจอร์) · Payout: 85% (จุดคุ้มทุน 54.05%)

---

## 1. สรุปผลการทดลอง (Executive Summary)

การทดลองนี้ทำการสร้างและเปรียบเทียบ **4 สถาปัตยกรรม Machine Learning** ที่แตกต่างกัน ภายใต้ Purged Walk-Forward 6 Folds เดียวกัน โดยใช้ Loss function แบบ EV-Weighted (Cost-Weighted) เพื่อมุ่งเป้าไปที่จุดคุ้มทุนของ Binary Options:
1. **LightGBM:** โมเดล GBDT ดั้งเดิมของโปรเจกต์
2. **XGBoost:** GBDT ด้วย Second-order split finding
3. **CatBoost:** GBDT ชนิด Oblivious/Symmetric Trees
4. **1D-CNN + BiLSTM (PyTorch):** Deep Neural Network ผสาน Conv1D + Bidirectional LSTM บน GPU (NVIDIA RTX 5070)

---

## 2. ตารางเปรียบเทียบผลลัพธ์ทั้ง 4 สถาปัตยกรรม

| สถาปัตยกรรมโมเดล | ประเภท | ฮาร์ดแวร์ | เวลาประมวลผล (วินาที) | จำนวนเทรด | Win Rate (%) | Wilson 95% LCB (%) | Net EV ต่อเทรด | ข้อสังเกตพฤติกรรม |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | GBDT (Asymmetric) | CPU Multi-core | **74.3s** (เร็วสุด) | 348 | 49.71% | 45.32% | -0.0803 | แม่นยำสูงสุดในกลุ่ม แต่ยังได้เพียง ~50% ในช่วง 2024–2026 |
| **XGBoost** | GBDT (Greedy) | CPU Multi-core | 91.4s | 376 | 47.34% | 43.14% | -0.1242 | ให้จำนวนเทรดใกล้เคียง LightGBM แต่ Win rate ต่ำกว่า |
| **CatBoost** | GBDT (Symmetric) | CPU Multi-core | 532.1s (~8.9 นาที) | 0 | - | - | - | Regularization เข้มมาก คะแนนความน่าจะเป็นไม่ผ่าน 54.05% จึงไม่มีไม้เทรด |
| **CNN + BiLSTM** | Deep Learning | GPU (RTX 5070) | 1,304.6s (~21.7 นาที)| 0 | - | - | - | โมเดลกระจาย Logits รอบ Base Rate (~50%) หลัง Calibrate จึงไม่มีไม้เทรดที่ข้าม Threshold |

---

## 3. การวิเคราะห์เชิงลึก (Quant Insights)

### 1. ยืนยันสมมติฐานเรื่อง Regime Shift หลังปี 2022
* ผลลัพธ์ของโมเดลบนช่วงข้อมูล 2024–2026 ยืนยันข้อสังเกตเดิมใน `docs/audit-gbpusd-k5.md` อย่างชัดเจน:
  - ในช่วงปี 2018–2021 สัญญาณเทคนิคอลยังมี Edge บ้าง (Win rate 54–58%)
  - แต่ตั้งแต่ปี 2022 เป็นต้นมา จนถึง 2024–2026 ตลาดมีความมีประสิทธิภาพ (Market Efficiency) สูงขึ้นมาก โมเดลทั้งแบบ Tree และ Deep Sequence ทำ Win rate นอกตัวอย่างได้เพียง **47% – 50%** (เปรียบเสมือนการโยนเหรียญ)
* สิ่งนี้สะท้อนว่า **ปัญหาไม่ได้อยู่ที่ตัวอัลกอริทึม ML** (เปลี่ยนจาก LightGBM เป็น XGBoost, CatBoost หรือ CNN+LSTM ก็ไม่ได้ช่วยให้ชนะ Noise ในข้อมูล M1 ได้)

### 2. ความเร็วและประสิทธิภาพ (Compute Efficiency)
* **LightGBM** ยังคงเป็น King of Tabular Data ในแง่ประสิทธิภาพ: เร็วกว่า CatBoost 7 เท่า และเร็วกว่า Neural Network เกือบ 18 เท่า โดยให้ค่า EV ดีที่สุด
* **CatBoost & CNN+LSTM** มีแนวโน้มที่จะ Conservative สูงมากบนข้อมูลการเงินที่มี Noise สูง ส่งผลให้ความน่าจะเป็นหลัง Platt Calibration ถูกบีบเข้าหา Prior Prob (~50%) และไม่ยอมส่งสัญญาณออกเลย (Zero Trades) ซึ่งในแง่หนึ่งคือการปฏิเสธที่จะเทรดในตลาดที่ไม่มี Edge

---

## 4. ก้าวต่อไป: สู่ Phase 3 (Reinforcement Learning)

เมื่อเราเห็นแล้วว่าโมเดลเดี่ยวๆ บนคู่เดียวไม่สามารถสร้าง Edge ทะลุ 54.05% ในช่วงหลังได้:
โจทย์ข้อ 3 ที่คุณตั้งไว้: **"นำ RL มาฝึก โดยให้ signal จาก ML ของทุกคู่เงินมาฝึก และ feature ยกเว้น raw price"** จะเข้ามาแก้ปัญหานี้โดยตรง:
- ให้ RL ทำหน้าที่เป็น **Portfolio Allocator / Meta-Controller**:
- RL ไม่ต้องพยายามทายแท่งเทียนทุกแท่ง แต่จะเรียนรู้ว่า **เมื่อไหร่ที่สัญญาณข้ามคู่เงิน (เช่น EUR, GBP, JPY, DXY) มีความสอดคล้องกันสูงพอ** จึงจะตัดสินใจ Execute ไม้เทรด หรือเลือกที่จะ "อยู่เฉยๆ (No Trade)" ในช่วงเวลาที่ความน่าจะเป็นต่ำ
