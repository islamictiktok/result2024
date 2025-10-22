import requests, time, io, os, random, datetime, traceback, re, json, csv
from bs4 import BeautifulSoup
from PIL import Image
import pandas as pd
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
# توحيد أسماء المواد
# ============================
def normalize_subject(name: str) -> str:
    s = re.sub(r"\s+", " ", name).strip()
    s = s.replace("الأحياء", "الاحياء")
    s = s.replace("الادب الانجليزى", "الأدب الإنجليزي")
    return s

# ============================
# دالة الاستخراج من HTML
# ============================
def parse_from_given_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    name = ""
    seat_no = ""
    result_text = ""
    percent = ""
    subjects = {}

    tables = soup.find_all("table")
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
                if re.fullmatch(r'\d{1,3}', score):
                    iv = int(score)
                    if 0 <= iv <= 100:
                        subjects[subj] = score

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
    for subj, score in subjects.items():
        row[subj] = score

    return row

# ============================
# بناء DataFrame وحفظ CSV
# ============================
def build_and_save_dynamic_csv(rows, out_path):
    df = pd.DataFrame(rows)
    base_cols = ["رقم الجلوس","اسم الطالب","النسبة","النتيجة"]
    extra_cols = [c for c in df.columns if c not in base_cols + ["__raw","error"]]
    final_cols = base_cols + sorted(extra_cols) + ["__raw","error"] if "error" in df.columns else base_cols + sorted(extra_cols) + ["__raw"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[final_cols]
    safe_write_csv(df, out_path)
    print(f"✅ Saved CSV: {out_path} | rows: {len(df)}")
    return df

# ============================
# الحلقة الرئيسية
# ============================
start_from = read_checkpoint() or START_SEAT
print("Starting from seat:", start_from)

session = new_session()
token = None
try:
    token = get_token(session)
except Exception as e:
    print("Warning: could not fetch verification token:", e)

