import cv2
import numpy as np
import mediapipe as mp

# ---------- AYARLAR (HARDCORE MOD) ----------
CAM_INDEX = 0
BG_FRAMES = 60

ALPHA_BG = 0.04          # Arka plan daha yavaş değişsin (stabilite için)
MASK_BLUR = 15           # Biraz daha fazla blur
MASK_THRESH = 0.45       # Daha agresif silme (gürültü istemiyoruz)
EDGE_FEATHER = 9         # Kenar yumuşatma arttırıldı

MASK_TEMPORAL = 0.85     # Çok yüksek stabilizasyon (titreme yok)
BASE_OPACITY = 0.08      # %8 Görünürlük (neredeyse yoksun)
MOTION_THRESH = 10.0     # Bu hareket değerini geçersen %0 olursun
# -------------------------------------------

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı.")

# Model selection 1 (Landscape) daha iyi çalışır
mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

bg = None
bg_count = 0
prev_mask = None
cloak_enabled = True

print("HAZIRLIK: İlk 60 karede ekrandan çık! (Arka plan öğreniliyor)")
print("Kontroller: 'c': Cloak Aç/Kapa | 'r': Reset BG | ESC: Çıkış")

# Dilation (genişletme) için kernel
dilate_kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_float = frame.astype(np.float32)

    if bg is None:
        bg = frame_float.copy()

    # ---------- 1. ARKA PLAN ÖĞRENME ----------
    if bg_count < BG_FRAMES:
        cv2.accumulateWeighted(frame_float, bg, 0.2)
        bg_count += 1
        
        # Yükleme ekranı
        vis = frame.copy()
        cv2.rectangle(vis, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        loading_w = int((bg_count / BG_FRAMES) * 400)
        cv2.rectangle(vis, (120, 220), (120 + loading_w, 260), (0, 255, 0), -1)
        cv2.putText(vis, "SISTEM YUKLENIYOR - KADRAJDAN CIK", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Ghost Terminal v2", vis)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    bg8 = bg.astype(np.uint8)

    # ---------- 2. HAREKET ANALİZİ (DYNAMIC OPACITY) ----------
    # Arka plan ile şu anki kare arasındaki fark (Motion)
    diff = cv2.absdiff(frame, bg8)
    motion_score = np.mean(diff)

    # Eğer çok hareket varsa (motion > 10), tamamen yok ol (0.0)
    # Değilse varsayılan hayalet modunda kal (0.08)
    if motion_score > MOTION_THRESH:
        current_opacity = 0.0
        status_sub = "MODE: ACTIVE CAMO (MOVING)"
    else:
        current_opacity = BASE_OPACITY
        status_sub = "MODE: GHOST (IDLE)"

    # ---------- 3. SEGMENTASYON ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_seg.process(rgb)
    mask = res.segmentation_mask

    # a) Maskeyi biraz temizle
    mask = cv2.GaussianBlur(mask, (MASK_BLUR | 1, MASK_BLUR | 1), 0)
    
    # b) Eşikleme (Threshold)
    mask = (mask > MASK_THRESH).astype(np.float32)

    # c) GÖLGE VE KOYU ALANLARI MASKEYE EKLEME (Kritik)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Çok koyu alanları (gölge gibi) maskeye dahil et ki orayı da arka planla boyasın
    shadow_mask = (gray < 60).astype(np.float32)
    # Gölge maskesini ana maskeye hafifçe yediriyoruz
    mask = np.clip(mask + 0.3 * shadow_mask, 0.0, 1.0)

    # d) DILATION (Genişletme) - Parmak aralarını kapatır
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # ---------- 4. TEMPORAL STABILIZATION (Titreşimi Kes) ----------
    if prev_mask is None:
        prev_mask = mask
    mask = MASK_TEMPORAL * prev_mask + (1 - MASK_TEMPORAL) * mask
    prev_mask = mask

    # e) FEATHER (Yumuşak Geçiş)
    mask = cv2.GaussianBlur(mask, (EDGE_FEATHER | 1, EDGE_FEATHER | 1), 0)
    mask3 = mask[..., None]

    # ---------- 5. ARKA PLAN GÜNCELLEME (Sinsi Mod) ----------
    # Sadece maskenin olmadığı (senin olmadığın) yerleri çok yavaş güncelle
    inv_mask = 1.0 - mask3
    bg = bg * (1 - ALPHA_BG * inv_mask) + frame_float * (ALPHA_BG * inv_mask)
    bg8 = bg.astype(np.uint8)

    # ---------- 6. CLOAK RENDER ----------
    if cloak_enabled:
        # Formül: (Sen * Opacity) + (Arka Plan * (1-Opacity))
        # Maskenin olduğu bölgeyi bu karışımla doldur
        cloak_area = (frame_float * current_opacity) + (bg.astype(np.float32) * (1 - current_opacity))
        
        # Son birleştirme
        out = (cloak_area * mask3 + frame_float * (1 - mask3)).astype(np.uint8)
    else:
        out = frame.copy()
        status_sub = "SYSTEM OFF"

    # ---------- 7. HUD (HEADS UP DISPLAY) ----------
    # Matrix yeşili arayüz
    cv2.putText(out, "GHOST TERMINAL v2.0", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    color_st = (0, 255, 0) if cloak_enabled else (0, 0, 255)
    cv2.putText(out, status_sub, (20, out.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_st, 2)

    # Hareket barı (Opsiyonel debug)
    cv2.rectangle(out, (out.shape[1]-20, out.shape[0]-10), 
                  (out.shape[1]-10, out.shape[0]-10 - int(motion_score*3)), (0, 255, 255), -1)

    cv2.imshow("Ghost Terminal v2", out)

    # ---------- KONTROLLER ----------
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    if key == ord('r'):
        bg = frame_float.copy()
        bg_count = 0
        prev_mask = None
    if key == ord('c'):
        cloak_enabled = not cloak_enabled

cap.release()
cv2.destroyAllWindows()