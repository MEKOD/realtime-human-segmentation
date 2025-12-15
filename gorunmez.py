import cv2
import numpy as np
import mediapipe as mp


# =========================
# CONFIG
# =========================
CAM_INDEX = 0
WARMUP_FRAMES = 60          # background capture frames (stay out of frame)
BG_UPDATE_ALPHA = 0.03      # background EMA update speed (small = stable)
SEG_MODEL = 1               # 0 = general, 1 = landscape (often better)

# Mask cleanup
MASK_BLUR = 13              # gaussian blur on raw mask
MASK_THRESH = 0.45          # threshold for person mask
DILATE_ITER = 1             # close finger gaps a bit
DILATE_KSIZE = 5
FEATHER_BLUR = 9            # soften edges

# Temporal stabilization
MASK_TEMPORAL = 0.85        # higher = less jitter, more lag (0..1)

# Cloak behavior
BASE_OPACITY = 0.08         # default visibility (0 = invisible, 1 = normal)
MOTION_TO_INVIS = 10.0      # if motion score exceeds -> opacity goes to 0
MOTION_SMOOTH = 0.8         # smooth motion score to avoid flicker (0..1)

# UI
WINDOW_NAME = "Cloak"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# HELPERS
# =========================
def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı. CAM_INDEX yanlış olabilir.")

    mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=SEG_MODEL)

    bg = None
    warmup = 0
    prev_mask = None
    cloak_enabled = True

    # motion smoothing (EMA)
    motion_ema = 0.0

    dilate_kernel = np.ones((DILATE_KSIZE, DILATE_KSIZE), np.uint8)

    print("Controls:  [c] cloak on/off   [r] recapture background   [esc] quit")
    print(f"Warmup: stay out of frame for first {WARMUP_FRAMES} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_f = frame.astype(np.float32)

        if bg is None:
            bg = frame_f.copy()

        # -------------------------
        # BACKGROUND WARMUP CAPTURE
        # -------------------------
        if warmup < WARMUP_FRAMES:
            cv2.accumulateWeighted(frame_f, bg, 0.20)
            warmup += 1

            vis = frame.copy()
            # minimal overlay
            cv2.putText(
                vis,
                f"Capturing background... {warmup}/{WARMUP_FRAMES} (step out)",
                (16, 34),
                FONT,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        # Convert background to uint8 for mixing/diff
        bg8 = bg.astype(np.uint8)

        # -------------------------
        # PERSON SEGMENTATION
        # -------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_seg.process(rgb)
        mask = res.segmentation_mask  # float [0..1] HxW

        # smooth raw mask
        mask = cv2.GaussianBlur(mask, (odd(MASK_BLUR), odd(MASK_BLUR)), 0)

        # threshold -> binary-ish
        mask = (mask > MASK_THRESH).astype(np.float32)

        # dilate to cover small gaps (fingers, edges)
        if DILATE_ITER > 0:
            mask = cv2.dilate(mask, dilate_kernel, iterations=DILATE_ITER).astype(np.float32)

        # temporal smoothing (EMA)
        if prev_mask is None:
            prev_mask = mask
        mask = (MASK_TEMPORAL * prev_mask + (1.0 - MASK_TEMPORAL) * mask).astype(np.float32)
        prev_mask = mask

        # feather edges
        mask = cv2.GaussianBlur(mask, (odd(FEATHER_BLUR), odd(FEATHER_BLUR)), 0)
        mask = np.clip(mask, 0.0, 1.0)
        mask3 = mask[..., None]

        # -------------------------
        # MOTION (ONLY INSIDE PERSON)
        # -------------------------
        # Use diff only where mask is present -> avoids background flicker driving opacity
        diff = cv2.absdiff(frame, bg8).astype(np.float32)
        # Convert diff to scalar motion score inside person area
        person_area = mask3
        denom = float(np.sum(mask) + 1e-6)
        motion_score = float(np.sum(np.mean(diff, axis=2) * mask) / denom)

        motion_ema = MOTION_SMOOTH * motion_ema + (1.0 - MOTION_SMOOTH) * motion_score

        # opacity rule
        current_opacity = 0.0 if motion_ema > MOTION_TO_INVIS else BASE_OPACITY
        current_opacity = clamp(current_opacity, 0.0, 1.0)

        # -------------------------
        # BACKGROUND UPDATE (ONLY NON-PERSON)
        # -------------------------
        inv = 1.0 - mask3
        bg = bg * (1.0 - BG_UPDATE_ALPHA * inv) + frame_f * (BG_UPDATE_ALPHA * inv)

        # -------------------------
        # RENDER
        # -------------------------
        if cloak_enabled:
            # Blend inside person mask:
            # out = mask*(opacity*frame + (1-opacity)*bg) + (1-mask)*frame
            cloak_mix = frame_f * current_opacity + bg * (1.0 - current_opacity)
            out = (cloak_mix * mask3 + frame_f * (1.0 - mask3)).astype(np.uint8)
        else:
            out = frame.copy()

        # -------------------------
        # MINIMAL STATUS OVERLAY
        # -------------------------
        status = "CLOAK ON" if cloak_enabled else "CLOAK OFF"
        cv2.putText(out, status, (16, out.shape[0] - 16), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # optional tiny debug (comment out if you want ultra-clean)
        cv2.putText(out, f"motion:{motion_ema:.1f}  opacity:{current_opacity:.2f}", (16, 60),
                    FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, out)

        # -------------------------
        # KEYS
        # -------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("c"):
            cloak_enabled = not cloak_enabled
        elif key == ord("r"):
            # recapture background
            bg = frame_f.copy()
            warmup = 0
            prev_mask = None
            motion_ema = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
