# -*- coding: utf-8 -*-
# كود كامل لاستخراج النتائج مباشرة من السورس HTML (مواد متغيرة) مع آلية الجلب والكابتشا والحفظ الدوري
# يعمل في Google Colab أو أي بيئة بايثون عادية

!apt-get update -y
!apt-get install -y tesseract-ocr libtesseract-dev -qq
!pip install --quiet pytesseract opencv-python-headless numpy beautifulsoup4 pandas

import requests, time, io, os, random, datetime, traceback, re, json, csv
from bs4 import BeautifulSoup
from PIL import Image
import pandas as pd
from google.colab import files
import pytesseract
import numpy as np
import cv2
from collections import Counter

# ============================
# إعدادات قابلة للتعديل
# ============================
BASE = "https://result.sd/"
CAPTCHA_URL = BASE + "captcha/image"
OUT_CSV = "results_clean.csv"
LOG_RAW = "logs_raw.txt"
CHECKPOINT = "last_seat.txt"

START_SEAT = 100
END_SEAT = 200

REQUEST_TIMEOUT = 120
MAX_RETRIES = 8
BACKOFF_INITIAL = 1.0
BACKOFF_FACTOR = 2.0
JITTER_MAX = 0.5
REFRESH_SESSION_AFTER = 20
RESOLVE_NEW_CAPTCHA_AFTER_CONSECUTIVE_SERVER_ERRORS = 2
BATCH_SAVE = 100
DELAY_BETWEEN = 0.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": BASE,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

# ============================
# وظائف مساعدة للملف والقراءة الآمنة
# ============================
def write_checkpoint(seat):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        f.write(str(seat))

def read_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            return int(open(CHECKPOINT, "r", encoding="utf-8").read().strip())
        except:
            return None
    return None

def safe_read_existing_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding='utf-8')
    except Exception:
        return pd.read_csv(path, engine='python', encoding='utf-8', on_bad_lines='skip', quotechar='"', escapechar='\\')

def safe_write_csv(df, path):
    df.to_csv(path, index=False, mode='w', header=True, quoting=csv.QUOTE_ALL, escapechar='\\', encoding='utf-8')

# ============================
# وظائف الشبكة والكابتشا
# ============================
def new_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def get_token(s):
    r = s.get(BASE, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    inp = soup.find("input", {"name":"__RequestVerificationToken"})
    return inp["value"] if inp else None

def download_captcha_bytes(s):
    r = s.get(CAPTCHA_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def post_search_raw(s, seat, token, captcha_text):
    data = {
        "SearchValue": str(seat),
        "CaptchaAnswer": str(captcha_text),
        "__RequestVerificationToken": token
    }
    r = s.post(BASE, data=data, headers={"Content-Type":"application/x-www-form-urlencoded","Referer":BASE}, timeout=REQUEST_TIMEOUT)
    return r

def log_raw(seat, status, text):
    with open(LOG_RAW, "a", encoding="utf-8") as f:
        f.write(f"--- {datetime.datetime.utcnow().isoformat()} --- seat:{seat} status:{status}\n")
        f.write(text[:5000] + "\n\n")

def safe_post_with_retries(s, seat, token, captcha_solution):
    attempt = 0
    backoff = BACKOFF_INITIAL
    while attempt < MAX_RETRIES:
        try:
            resp = post_search_raw(s, seat, token, captcha_solution)
            if resp.status_code == 200:
                return resp
            if 500 <= resp.status_code < 600:
                log_raw(seat, resp.status_code, resp.text)
                raise requests.exceptions.HTTPError(f"Server {resp.status_code}")
            return resp
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            attempt += 1
            jitter = random.uniform(0, min(JITTER_MAX, backoff * 0.5))
            sleep_for = backoff + jitter
            print(f"[retry] seat {seat} attempt {attempt}/{MAX_RETRIES} failed: {e}. backoff {sleep_for:.1f}s")
            time.sleep(sleep_for)
            backoff *= BACKOFF_FACTOR
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts for seat {seat}")

def solve_captcha_advanced(img_bytes, display_image=False):
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (1,1), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enlarged = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(enlarged, config="--psm 8 -c tessedit_char_whitelist=0123456789")
    digits = ''.join(filter(str.isdigit, text))
    return digits.strip()

# ============================
# توحيد أسماء المواد (تنظيف وتصحيحات بسيطة)
# ============================
def normalize_subject(name: str) -> str:
    s = re.sub(r"\s+", " ", name).strip()
    # تصحيحات شائعة حسب البيانات
    s = s.replace("الأحياء", "الاحياء")
    s = s.replace("الادب الانجليزى", "الأدب الإنجليزي")
    # لو عايز تصحح "الرياضيات المنخصصة" إلى صيغة أخرى، فعّل السطر التالي:
    # s = s.replace("الرياضيات المنخصصة", "الرياضيات المتخصصة")
    return s

# ============================
# دالة الاستخراج من HTML الناتج مباشرة (مواد متغيرة)
# ============================
def parse_from_given_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    name = ""
    seat_no = ""
    result_text = ""
    percent = ""
    subjects = {}

    tables = soup.find_all("table")
    # الجدول الأول: بيانات عامة
    if tables:
        first_table = tables[0]
        for tr in first_table.find_all("tr"):
            cols = tr.find_all("td")
            if len(cols) >= 2:
                key = re.sub(r"\s+", " ", cols[0].get_text(strip=True)).strip()
                val = cols[1].get_text(strip=True)
                if "اسم" in key:
                    name = val
                elif "جلوس" in key or "رقم الجلوس" in key:
                    seat_no = val
                elif "نتيجة" in key:
                    result_text = val
                elif "نسبة" in key:
                    percent = val

    # المواد: من الجداول التالية فقط، نقبل أي مادة بدرجة رقمية 0..100
    non_subject_keys = {"اسم الطالب", "رقم الجلوس", "النتيجة", "النسبة"}
    for t in tables[1:]:
        for tr in t.find_all("tr"):
            cols = tr.find_all("td")
            if len(cols) >= 2:
                subj_raw = cols[0].get_text(strip=True)
                score = cols[1].get_text(strip=True)
                subj = normalize_subject(subj_raw)

                if not subj or subj in non_subject_keys:
                    continue

                # درجة رقمية فقط 0..100
                if re.fullmatch(r'\d{1,3}', score):
                    iv = int(score)
                    if 0 <= iv <= 100:
                        subjects[subj] = score

    # __raw: نص مختصر ومُنظف من الفوتر (اختياري)
    raw_text = soup.get_text(" ", strip=True)
    raw_text = re.sub(r"جميع الحقوق محفوظة.*$", "", raw_text).strip()
    raw_json = json.dumps(raw_text[:2000], ensure_ascii=False)

    row = {
        "رقم الجلوس": seat_no,
        "اسم الطالب": name,
        "النتيجة": result_text,
        "النسبة": percent,
        "__raw": raw_json
    }
    # إضافة المواد كأعمدة بأسمائها المتغيرة
    for subj, score in subjects.items():
        row[subj] = score

    return row

# ============================
# بناء DataFrame ديناميكي وحفظ CSV
# ============================
def build_and_save_dynamic_csv(rows, out_path):
    df = pd.DataFrame(rows)
    base_cols = ["رقم الجلوس","اسم الطالب","النسبة","النتيجة"]
    extra_cols = [c for c in df.columns if c not in base_cols + ["__raw","error"]]
    final_cols = base_cols + sorted(extra_cols) + ["__raw","error"] if "error" in df.columns else base_cols + sorted(extra_cols) + ["__raw"]
    # تأكد من وجود الأعمدة حتى لو فاضية
    for col in final_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[final_cols]
    safe_write_csv(df, out_path)
    print(f"✅ Saved CSV: {out_path} | rows: {len(df)}")
    return df

# ============================
# الحلقة الرئيسية: جلب، تحليل HTML، حفظ دوري
# ============================
start_from = read_checkpoint() or START_SEAT
print("Starting from seat:", start_from)

session = new_session()
token = None
try:
    token = get_token(session)
except Exception as e:
    print("Warning: could not fetch verification token:", e)

# حل كابتشا مبدئي
try:
    img = download_captcha_bytes(session)
    captcha_solution = solve_captcha_advanced(img, display_image=False)
    print("Captcha solved:", captcha_solution)
except Exception as e:
    captcha_solution = ""
    print("Captcha step skipped/failed:", e)

all_rows = []
saved_total = 0
request_count_in_session = 0
consecutive_server_errors = 0

seat = start_from
while seat <= END_SEAT:
    try:
        # تحديث توكن دوري
        try:
            new_token = get_token(session)
            if new_token:
                token = new_token
        except Exception:
            pass

        resp = safe_post_with_retries(session, seat, token, captcha_solution)
        status = resp.status_code
        text = resp.text

        if status == 200:
            parsed = parse_from_given_html(text)
            if not parsed.get("رقم الجلوس"):
                parsed["رقم الجلوس"] = str(seat)
            all_rows.append(parsed)
            print(f"OK {seat} -> {parsed.get('اسم الطالب','-')} | {parsed.get('النسبة','-')}")
            consecutive_server_errors = 0
            write_checkpoint(seat + 1)
        else:
            log_raw(seat, status, text)
            all_rows.append({"رقم الجلوس": str(seat), "اسم الطالب":"", "النسبة":"", "__raw": json.dumps(text[:1000], ensure_ascii=False), "error": f"status_{status}"})
            write_checkpoint(seat + 1)

    except Exception as e:
        print(f"ERR {seat} -> {e}")
        log_raw(seat, "exception", traceback.format_exc())
        all_rows.append({"رقم الجلوس": str(seat), "اسم الطالب":"", "النسبة":"", "__raw": json.dumps(str(e), ensure_ascii=False), "error": str(e)})
        consecutive_server_errors += 1

    # حفظ دوري ديناميكي بالأعمدة المتغيرة
    if len(all_rows) >= BATCH_SAVE:
        existing_df = safe_read_existing_csv(OUT_CSV)
        old_rows = existing_df.to_dict(orient='records') if existing_df is not None else []
        merged_rows = old_rows + all_rows
        final_df = build_and_save_dynamic_csv(merged_rows, OUT_CSV)
        saved_total = len(merged_rows)
        print(f"Saved {saved_total} rows to {OUT_CSV} (full rewrite)")
        all_rows = []

    request_count_in_session += 1

    # تحديث سيشن وكابتشا دوريًا
    if consecutive_server_errors >= RESOLVE_NEW_CAPTCHA_AFTER_CONSECUTIVE_SERVER_ERRORS or request_count_in_session >= REFRESH_SESSION_AFTER:
        print("Refreshing session and solving new captcha...")
        try:
            session = new_session()
            token = get_token(session)
            img = download_captcha_bytes(session)
            captcha_solution = solve_captcha_advanced(img, display_image=False)
            print("New captcha solved:", captcha_solution)
        except Exception as e:
            print("Session refresh failed:", e)
        consecutive_server_errors = 0
        request_count_in_session = 0

    seat += 1
    if DELAY_BETWEEN:
        time.sleep(DELAY_BETWEEN)

# حفظ المتبقي وإعادة كتابة الملف الكامل بأعمدة ديناميكية ثابتة
if all_rows:
    existing_df = safe_read_existing_csv(OUT_CSV)
    old_rows = existing_df.to_dict(orient='records') if existing_df is not None else []
    merged_rows = old_rows + all_rows
    final_df = build_and_save_dynamic_csv(merged_rows, OUT_CSV)
    saved_total = len(merged_rows)

print("✅ Finished. Total rows saved:", saved_total)

# تنزيل الملف في Colab
try:
    files.download(OUT_CSV)
except Exception:
    print(f"CSV written to {OUT_CSV}. Download not available in this environment.")
