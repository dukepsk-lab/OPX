# เอกสารปิดโครงการ OPX (Project Closure & Final Post-Mortem)

**วันที่ปิดโครงการ:** 26 ตุลาคม 2026  
**สถานะโครงการ:** **CLOSED / ARCHIVED**  
**คำตัดสินขั้นเด็ดขาด:** **STRICT NO-GO FOR REAL TRADING (ไม่อนุญาตให้นำเงินจริงไปเทรดในทุกกรณี)**  

---

## 1. บทสรุปภาพรวมโครงการ (Executive Summary)

โครงการ **OPX** ถูกจัดตั้งขึ้นเพื่อทำการวิจัยและพิสูจน์เชิงวิทยาศาสตร์ข้อมูลขั้นสูง (Scientific Quant Research) ว่า:
> *"เป็นไปได้หรือไม่ที่จะสร้างระบบ Machine Learning หรือ Reinforcement Learning ที่สามารถทำนายทิศทางราคา FX ในกรอบเวลาสั้น (M1/M5/M15) บนสัญญา Fixed-Time Trades / Binary Options จนมี Edge ชนะจุดคุ้มทุน (Breakeven Hurdle) $p^* = \frac{1}{1 + 0.85} \approx 54.05\%$ ได้อย่างยั่งยืน โดยไม่พึ่งพา Martingale และผ่านเกณฑ์การทดสอบทางสถิติที่ปราศจาก Data Snooping / Leakage?"*

หลังจากการทดสอบรอบด้าน ทั้งแบบ Baseline 15 Setup, การขยาย Feature Expansion 11 รูปแบบ, การทดสอบโมเดล ML สมัยใหม่ 4 ตระกูล, และการฝึก Reinforcement Learning Meta-Agent บน 5 คู่เงิน:
**คำตอบคือ: เป็นไปไม่ได้ในเชิงปฏิบัติการ (Infeasible for Production Trading)**  
ระบบทั้งหมดปฏิเสธการเทรด (NO-GO) ตามเกณฑ์ทางสถิติอย่างเคร่งครัด ซึ่งเป็นเครื่องยืนยันว่าโครงสร้างระบบระเบียบการทดสอบ (Validation Pipeline) สามารถตรวจจับและป้องกันความสูญเสียทางการเงินจริงได้อย่างสมบูรณ์แบบ

---

## 2. ลำดับการวิจัยและการทดลอง (Research Trajectory)

### ระยะที่ 1: Baseline Architecture & The 15 Setups (NO-GO)
- **ข้อมูล:** Dukascopy M1 Historical Bars ราว 11.8 ปี (2014–2026) ครอบคลุม 5 คู่เงินหลัก (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD) รวมกว่า 20 ล้านบาร์
- **สถาปัตยกรรม:**
  - Forward Expiry Labeling: Entry ที่ Open[t+1], Expiry ที่ Close[t+k], กรณีเสมอ (Tie) นับเป็นแพ้ตามกฎเกณฑ์ Broker
  - Purged Walk-Forward CV (6 folds) พร้อม Embargo ป้องกัน Information Leakage
  - LightGBM ควบคู่กับ EV-Weighted Custom Loss Function ถ่วงน้ำหนักตาม Payout 85%
  - Out-of-Sample Platt Calibration
  - Multi-Horizon Selection: $k \in \{1, 5, 15\}$ นาที
- **สิ่งที่ค้นพบ:**
  - รอบแรกเกิด False Positive ในคู่ EURUSD และ GBPUSD k=15 แต่การทำ In-depth Audit พบว่าเกิดจาก **New York Rollover Spreads (16:00–18:59 NY) และวันอาทิตย์** ที่สภาพคล่องเหือดแห้งและ Spread ถ่าง
  - หลังกรองช่วง Illiquid ออก ผลลัพธ์ทั้ง 15 Setup กลายเป็น **NO-GO 100%** ไม่มีคู่ใดผ่าน Gate ทางสถิติ 8 ข้อ

---

### ระยะที่ 2: Feature Engineering & Ablation Study (Phase 1 Deep-Dive)
- **วัตถุประสงค์:** ขยายคุณลักษณะของข้อมูลเพื่อดึงสัญญาณทางสถิติที่ลึกขึ้น ได้แก่:
  1. Real-time DXY (US Dollar Index M1 จาก Dukascopy 2.72 ล้านบาร์)
  2. Cross-Asset Relative Returns (5 คู่เงินเทียบตะกร้า USD)
  3. Volatility Squeeze Dynamics (Bollinger Bands vs Keltner/Realized Volatility)
  4. Fractional Differentiation ($d=0.35$ คง Stationarity ควบคู่ Memory)
- **การทดลอง:** ทดสอบ 11 รูปแบบบน GBPUSD $k=5$ (3.28 ล้านบาร์)
- **สิ่งที่ค้นพบ:**
  - **`squeeze` คือ Feature ที่ดีที่สุดของโครงการ:** ทุกการตั้งค่าที่มี Squeeze ให้ค่า Expected Value เป็นบวก และ Win Rate ชนะ Breakeven (54.19% – 54.93%) โดยมี PBO ต่ำมาก (0.04 – 0.10)
  - **`dxy + cross` เป็นตัวกรองระดับ Macro คุณภาพสูง:** ที่ Buffer 0.06 ทำ Win Rate ได้ถึง **65.96%** แต่กรองจนเหลือเพียง 47 ไม้ตลอด 8 ปี

---

### ระยะที่ 3: Multi-Model Architecture Benchmark (Phase 2 Deep-Dive)
- **วัตถุประสงค์:** ตรวจสอบว่าโมเดลระดับ State-of-the-Art อื่นๆ สามารถดึง Alpha ที่ LightGBM พลาดไปได้หรือไม่
- **โมเดลที่ทดสอบ:**
  1. **LightGBM:** Fast Gradient Boosted Trees (Baseline)
  2. **XGBoost:** Exact Second-order Approximation
  3. **CatBoost:** Symmetric Oblivious Trees
  4. **1D-CNN + BiLSTM (PyTorch / GPU RTX 5070):** Deep Temporal Representation
- **ผลการทดสอบ (บนข้อมูล 1,000,000 บาร์ช่วงปี 2024–2026):**
  - XGBoost: Win Rate 47.34% (Overfit และแย่กว่า Baseline)
  - LightGBM: Win Rate 49.71%
  - CatBoost และ 1D-CNN+BiLSTM: ไม่มีการออกเทรดเลย (0 Trades) เนื่องจากโมเดล Conservative สูง เมื่อผ่าน Platt Calibration แล้วไม่มีแท่งใดที่มีความมั่นใจเกิน 54.05%
- **บทเรียน:** ในตลาด M1 ช่วงปีหลังๆ พฤติกรรมราคาเข้าใกล้ Pure Martingale การเปลี่ยนสถาปัตยกรรมโมเดล ML ไม่สามารถเปลี่ยน Random Noise ให้เป็น Signal ได้

---

### ระยะที่ 4: Multi-Pair Reinforcement Learning Meta-Agent (Phase 3 Deep-Dive)
- **วัตถุประสงค์:** สร้าง RL Agent ทำหน้าที่เป็น "Portfolio Gatekeeper" คอยตัดสินใจว่าจะถือเงินสด (HOLD) หรือจะกด Call/Put ในคู่เงินใด โดยรับ State 150 มิติที่ไม่รวม Raw Price
- **สถาปัตยกรรม:**
  - Environment: `MultiPairBinaryOptionsEnv` (Gymnasium, Discrete(11) Action Space)
  - State: 150 Normalized Indicators + Cross-Pair Correlations + Base ML Probability Signals (ไม่มี Open/High/Low/Close)
  - Algorithm: PPO (Proximal Policy Optimization) 100,000 Timesteps
- **ผลการทดสอบ Out-of-Sample (บน 47,801 บาร์ที่ไม่เคยเห็น):**
  - Trades Taken: 8,012 ไม้
  - Win Rate: **45.16%** (แพ้ Breakeven 54.05% ขาดลอย)
  - Net PnL: **-1,493.50 R**
  - พฤติกรรมผิดปกติ: เกิด **Action Collapse** — Agent สลับแทงเฉพาะ USDCAD ถึง 52% ของเวลาทั้งหมด และแทบไม่เทรดคู่อื่น
- **บทเรียน:** กฎ *Garbage In, Garbage Out* — เมื่อ Base Signals มีความแม่นยำต่ำกว่า 50% และ Reward Structure มีความเสียเปรียบ (กำไร +0.85 แต่ขาดทุน -1.00) นโยบาย RL จะ Over-trade และพังทลายลงในที่สุด

---

## 3. สรุปบทเรียนทางวิทยาศาสตร์การเงิน (Scientific Insights)

1. **คณิตศาสตร์ของ Binary Options มี House Edge สูงมาก:**
   - ที่ Payout 85% ระบบต้องการ Win Rate ขั้นต่ำ 54.05% เพื่อ "เท่าทุน"
   - ในตลาด FX อัตรากำไร (Edge) ของสถาบันระดับ High Frequency Trading (HFT) มักอยู่ที่ 50.5% – 52.0% การเรียกร้องให้อัลกอริทึมทำนายแท่งเทียน 1 นาทีให้แม่นยำ $> 54.05\%$ ต่อเนื่องนับหมื่นไม้นั้น ฝืนทฤษฎี Efficient Market Hypothesis (EMH)
2. **อย่าใช้ Deep Learning และ RL ในสภาพแวดล้อม Low Signal-to-Noise:**
   - ยิ่งโมเดลมี Parameters มาก (Neural Networks, PPO) ยิ่งดูดกลืน Noise เข้าไปใน Weight กลายเป็น Overfitting ร้ายแรง
3. **Alpha เดียวที่จับต้องได้คือ Market Regimes ไม่ใช่ Continuous Directional Prediction:**
   - การพยายามทายว่า "อีก 5 นาทีจะขึ้นหรือลง" ในทุกๆ แท่ง เป็นไปไม่ได้
   - แต่การรอให้ตลาดเกิด **Volatility Squeeze (ความผันผวนบีบอัด)** แล้วเทรดตามทิศทาง Breakout ร่วมกับการยืนยันของ **Dollar Index (DXY)** เป็นสิ่งเดียวที่ให้ Win Rate สูงกว่า 54% อย่างมีนัยสำคัญทางสถิติ

---

## 4. สถานะของ Codebase และความสมบูรณ์ของระบบ

- **Unit & Integration Tests:** 75 ผ่านทั้งหมด (100% Pass)
- **ความปลอดภัยของระบบ:**
  - ปราศจาก Data Leakage
  - ปราศจาก Lookahead Bias
  - กรอง Illiquid Rollover Hours ออกเรียบร้อย
  - ห้ามระบบ Martingale และปฏิเสธการเชื่อมต่อกับ OTC Data อย่างเด็ดขาด
- **รายงานการวิจัยทั้งหมดในคลังเอกสาร:**
  - [`docs/full-report.md`](file:///c:/Users/swing/Desktop/TRADING/OPX/docs/full-report.md) — รายงาน Baseline 15 Setups
  - [`docs/ablation-report-gbpusd-k5.md`](file:///c:/Users/swing/Desktop/TRADING/OPX/docs/ablation-report-gbpusd-k5.md) — รายงาน Feature Ablation Study
  - [`docs/model-benchmark-report-gbpusd-k5.md`](file:///c:/Users/swing/Desktop/TRADING/OPX/docs/model-benchmark-report-gbpusd-k5.md) — รายงานเปรียบเทียบ Alternative ML Models
  - [`docs/rl-meta-agent-report.md`](file:///c:/Users/swing/Desktop/TRADING/OPX/docs/rl-meta-agent-report.md) — รายงานการทดลอง Reinforcement Learning Meta-Agent
  - [`docs/audit-report-eurusd-k15.md`](file:///c:/Users/swing/Desktop/TRADING/OPX/docs/audit-report-eurusd-k15.md) — รายงานการตรวจสอบ False Positive จาก Rollover Hours

---

## 5. คำแถลงปิดโครงการ (Final Sign-Off)

โครงการ **OPX** ถือว่าบรรลุวัตถุประสงค์ของการวิจัยเชิงปริมาณ (Quantitative Research) อย่างสมบูรณ์แบบ แม้ผลลัพธ์จะปฏิเสธการเทรด แต่การที่ระบบสามารถพิสูจน์ความจริงเชิงประจักษ์ ยืนหยัดในระเบียบวิธีทางสถิติ และปฏิเสธไม่ให้นำเงินทุนไปเสี่ยงในระบบที่ไม่มีแต้มต่อ ถือเป็นความสำเร็จสูงสุดของระบบความปลอดภัยและการวิจัย

**สถานะ: ปิดโครงการและเก็บเข้าคลังข้อมูลวิจัย (Archived)**
