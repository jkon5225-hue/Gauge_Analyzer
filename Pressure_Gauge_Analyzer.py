import os
import re
import shutil
import warnings
import cv2
import numpy as np
import easyocr

# 警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

# --- 出力用フォルダの確認・初期化 ---
OUTPUT_DIR = "output"

def initialize_output_dir(output_dir: str = OUTPUT_DIR) -> None:
    """出力用フォルダの確認・初期化（既存の場合は中身をすべて削除）"""
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"ファイル削除エラー ({file_path}): {e}")
        print(f"既存の '{output_dir}' フォルダの中身をクリアしました。")
    else:
        os.makedirs(output_dir)
        print(f"'{output_dir}' フォルダを新しく作成しました。")


def detect_meter_circle(img_gray: np.ndarray) -> tuple[float, float, float]:
    """ハフ変換を用いてメーター枠（円）を検出し、中心座標 (cx, cy) と半径 r を返す"""
    height, width = img_gray.shape[:2]
    blurred = cv2.medianBlur(img_gray, 5)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=390,
        param2=50,
        minRadius=int(width / 3),
        maxRadius=width,
    )

    if circles is not None:
        circles_arr = np.round(circles[0]).astype(int)
        cx, cy, r1 = map(float, circles_arr[0])
        return cx, cy, r1
    
    return float(width / 2), float(height / 2), float(width / 3)


def analyze_bourdon_gauge_geometric(
    image_path: str = "out_sample2.jpg",
    kmn: float = -0.1,
    kmx: float = 0.1,
    save_debug_images: bool = True,
) -> float:
    """真上の値を (kmn + kmx) / 2 と定義し、信頼性の高いラインと真上基準線の交点からセンターを補正して圧力値を算出する関数"""
    img2 = cv2.imread(image_path, 1)
    img_gray = cv2.imread(image_path, 0)

    if img2 is None or img_gray is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    height, width = img_gray.shape[:2]
    cx, cy, r1 = detect_meter_circle(img_gray)

    yohaku = 10
    x1 = max(0, int(cx - r1 - yohaku))
    y1 = max(0, int(cy - r1 - yohaku))
    x2 = min(width, int(cx + r1 + yohaku))
    y2 = min(height, int(cy + r1 + yohaku))

    cropped_img = img2[y1:y2, x1:x2].copy()
    cropped_gray = img_gray[y1:y2, x1:x2].copy()
    ccx, ccy = float(cx - x1), float(cy - y1)

    if save_debug_images:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "out_sample2.jpg"), cropped_img)

    # 2. 二値化とエッジ検出
    _, img_thresh = cv2.threshold(cropped_gray, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img_thresh, 50, 200, apertureSize=3)

    # 3. ハフ直線検出
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 35)
    if lines is None or len(lines) < 2:
        raise ValueError("直線を検出できませんでした。")

    upper_lines = []
    lower_lines = []
    c_height, c_width = cropped_img.shape[:2]

    for line in lines[:5]:
        for rho, theta in line:
            a_val = np.cos(theta)
            b_val = np.sin(theta)
            x0 = a_val * rho
            y0 = b_val * rho

            x1_l = int(x0 + 1000 * (-b_val))
            y1_l = int(y0 + 1000 * (a_val))
            x2_l = int(x0 - 1000 * (-b_val))
            y2_l = int(y0 - 1000 * (a_val))

            dx = x2_l - x1_l
            if dx == 0:
                dx = 1e-6
            a0 = (y2_l - y1_l) / dx
            b0 = y1_l - a0 * x1_l

            line_mask = np.zeros_like(edges)
            cv2.line(line_mask, (x1_l, y1_l), (x2_l, y2_l), 255, thickness=1)
            support_count = cv2.countNonZero(cv2.bitwise_and(edges, line_mask))

            y_at_center = a0 * ccx + b0
            if y_at_center < ccy:
                upper_lines.append((rho, theta, x1_l, y1_l, x2_l, y2_l, support_count, a0, b0))
            else:
                lower_lines.append((rho, theta, x1_l, y1_l, x2_l, y2_l, support_count, a0, b0))

    if not upper_lines:
        upper_lines.append((lines[0][0][0], lines[0][0][1], 0, int(ccy), c_width, int(ccy), 0, 0.0, ccy))
    if not lower_lines:
        lower_lines.append((lines[1][0][0], lines[1][0][1], 0, int(ccy + 10), c_width, int(ccy + 10), 0, 0.0, ccy + 10))

    best_upper = max(upper_lines, key=lambda x: x[6])
    best_lower = max(lower_lines, key=lambda x: x[6])

    chosen_line = best_upper if best_upper[6] >= best_lower[6] else best_lower

    _, _, ux1, uy1, ux2, uy2, _, _, _ = best_upper
    _, _, lx1, ly1, lx2, ly2, _, _, _ = best_lower
    _, _, _, _, _, _, _, ca, cb = chosen_line

    # センター位置の再導出
    ccy = ca * ccx + cb
    print(f"センター座標を補正しました: y={ccy:.1f}")

    # 真上基準線の描画
    cv2.line(cropped_img, (int(ccx), int(ccy)), (int(ccx), 0), (0, 255, 255), 2)

    if best_upper[6] > best_lower[6]:
        cv2.line(cropped_img, (int(ux1), int(uy1)), (int(ux2), int(uy2)), (0, 0, 255), 1)
        vx, vy = float(ux2 - ux1), float(uy2 - uy1)
    else:
        cv2.line(cropped_img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 0, 255), 1)
        vx, vy = float(lx2 - lx1), float(ly2 - ly1)

    if vx > 0:
        vx, vy = -vx, -vy

    norm = np.hypot(vx, vy)
    if norm > 1e-6:
        vx, vy = vx / norm, vy / norm
    else:
        vx, vy = -1.0, 0.0

    dot_prod = vx * 0.0 + vy * (-1.0)
    angle_from_zero = -float(np.arccos(np.clip(dot_prod, -1.0, 1.0)) * 180.0 / np.pi)

    mid_value = (kmn + kmx) / 2.0
    span = kmx - kmn
    full_scale_angle = 133.0
    pressure_offset = (angle_from_zero / full_scale_angle) * (span / 2.0)

    atai = np.clip(mid_value + pressure_offset, kmn, kmx)

    # 偏角の方向に緑線を描画
    rad_green = np.radians(angle_from_zero)
    gx2 = int(ccx + (r1 * 0.9) * np.sin(rad_green))
    gy2 = int(ccy - (r1 * 0.9) * np.cos(rad_green))
    cv2.line(cropped_img, (int(ccx), int(ccy)), (gx2, gy2), (0, 255, 0), 2)

    print(f"真上の基準値: {mid_value:.3f} MPa, 偏角: {angle_from_zero:.2f}° , 最終解析圧力: {atai:.3f} MPa")

    if save_debug_images:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "meter_line.jpg"), cropped_img)

    return atai


def extract_kmx_kmn(image_path: str) -> tuple[float | None, float | None]:
    """ブルドン管圧力計・連成計から最小値(kmn)と最大値(kmx)を抽出する関数"""
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        img = cv2.imread(image_path)
        if img is None:
            print(f'画像を読み込めません: {image_path}')
            return None, None

        img_height, img_width = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ccx, ccy, r1 = detect_meter_circle(img_gray)

        # 赤色マスクの作成
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
        )

        results = []
        results.extend(reader.readtext(img, allowlist='-0123456789.'))

        img_resized = cv2.resize(img_gray, (0, 0), fx=2.0, fy=2.0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(img_resized)

        for bbox, text, prob in reader.readtext(enhanced, allowlist='-0123456789.'):
            scaled_bbox = [[pt[0] / 2.0, pt[1] / 2.0] for pt in bbox]
            results.append((scaled_bbox, text, prob))

        num_pattern = re.compile(r'^-?\d+(?:\.\d+)?$')
        raw_candidates = []

        for bbox, text, prob in results:
            cleaned_text = text.strip().replace(' ', '').replace('O', '0').replace('o', '0').replace(',', '.')
            if not num_pattern.fullmatch(cleaned_text) or prob < 0.30:
                continue

            try:
                val = float(cleaned_text)
                pts = np.array(bbox, dtype=np.int32)
                x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
                x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])

                box_cx, box_cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                distance = np.hypot(box_cx - ccx, ccy - box_cy)
                angle = np.degrees(np.arctan2(box_cx - ccx, ccy - box_cy))

                sy_min, sy_max = max(0, y_min - 5), min(img_height, y_max + 5)
                sx_min, sx_max = max(0, x_min - 5), min(img_width, x_max + 5)
                red_pixel_count = np.sum(red_mask[sy_min:sy_max, sx_min:sx_max] > 0)

                if val > 0 and box_cx < ccx and red_pixel_count > 5:
                    val = -val

                if 0.2 * r1 <= distance <= 1.4 * r1 and abs(angle) >= 30.0:
                    raw_candidates.append((val, angle, y_max - y_min, distance))
            except (ValueError, TypeError, IndexError):
                continue

        if not raw_candidates:
            return None, None

        distances = [c[3] for c in raw_candidates]
        median_distance = np.median(distances)
        filtered_by_distance = [c for c in raw_candidates if median_distance * 0.75 <= c[3] <= median_distance * 1.25]

        if not filtered_by_distance:
            return None, None

        max_height = max(c[2] for c in filtered_by_distance)
        candidates = [c for c in filtered_by_distance if c[2] >= max_height * 0.50]

        if not candidates:
            return None, None

        rates = [val / angle for val, angle, *_ in candidates if abs(angle) > 45.0]
        if not rates:
            return None, None

        avg_rate = np.median(rates)
        calc_kmn, calc_kmx = avg_rate * (-135.0), avg_rate * 135.0

        if not np.isfinite(calc_kmn) or not np.isfinite(calc_kmx) or calc_kmx <= 0:
            return None, None

        raw_min = min(c[0] for c in candidates)
        raw_max = max(c[0] for c in candidates)

        kmn = -0.1 if (raw_min < 0 and calc_kmn < -0.05) else 0.0
        
        if calc_kmx <= 0.15: kmx = 0.1
        elif calc_kmx <= 0.25: kmx = 0.2
        elif calc_kmx <= 0.35: kmx = 0.3
        elif calc_kmx <= 0.6: kmx = 0.5
        else: kmx = round(raw_max, 1)

        return kmn, kmx

    except Exception as e:
        print(f'スケール抽出エラー: {e}')
        return None, None


def rcog(image: np.ndarray) -> tuple[list | None, str | None, float]:
    """画像から数字を読み取り、最も信頼度の高いテキストと信頼度を返す"""
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image, detail=1, allowlist='0123456789.')
    if results:
        best_result = max(results, key=lambda x: x[2])
        return best_result[0], best_result[1], best_result[2]
    return None, None, 0.0


def extract_kmx_kmn2(image_path: str) -> tuple[float, float]:
    """フォールバック用スケール抽出関数（特定領域を個別にスキャン）"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    img_height, img_width = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ccx, ccy, r1 = detect_meter_circle(img_gray)

    print("【フォールバック解析開始】個別領域スキャンを実行します。")
    threshold_levels = [70, 80, 110]
    kmn, kmx = 0.0, 70.0

    # 右下領域 (MAX候補)
    rb_roi = img_gray[int(ccy):int(ccy + r1), int(ccx):int(ccx + r1)]
    if rb_roi.size > 0:
        for th in threshold_levels:
            _, thresh_roi = cv2.threshold(rb_roi, th, 255, cv2.THRESH_BINARY)
            resized_roi = cv2.resize(thresh_roi, (0, 0), fx=2.0, fy=2.0)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"fallback_right_bottom_thresh_{th}.jpg"), resized_roi)
            _, text, _ = rcog(resized_roi)
            if text:
                try:
                    val = float(text)
                    if val < kmx:
                        kmx = val
                except ValueError:
                    pass

    # 左下領域 (MIN候補)
    lb_roi = img_gray[int(ccy):int(ccy + r1), int(ccx - r1):int(ccx)]
    if lb_roi.size > 0:
        for th in threshold_levels:
            _, thresh_roi = cv2.threshold(lb_roi, th, 255, cv2.THRESH_BINARY)
            resized_roi = cv2.resize(thresh_roi, (0, 0), fx=2.0, fy=2.0)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"fallback_left_bottom_thresh_{th}.jpg"), resized_roi)
            _, text, _ = rcog(resized_roi)
            if text:
                try:
                    val = float(text)
                    if val < kmx:
                        kmn = val
                except ValueError:
                    pass

    return float(kmn), float(kmx)


# --- 実行エントリポイント ---
if __name__ == "__main__":
    initialize_output_dir(OUTPUT_DIR)
    image_file = "moto meter.jpg"

    try:
        kmn, kmx = extract_kmx_kmn(image_file)

        if kmn is None or kmx is None or kmn > kmx or kmx > 70:
            print("STEP1では適切なスケールを認識できませんでした。フォールバック処理を実行します。")
            kmn, kmx = extract_kmx_kmn2(image_file)

        print(f"検出されたスケール範囲: 最小({kmn}) 〜 最大({kmx})")

        pressure = analyze_bourdon_gauge_geometric(
            image_file, kmn=kmn, kmx=kmx, save_debug_images=True
        )
        print(f"解析された計測圧力: {pressure:.3f} MPa")

    except Exception as e:
        print(f"エラーが発生しました: {e}")