import io
import os
import cv2
import math
import uuid
import base64
import numpy as np
import pandas as pd
import easyocr
import re
from rapidfuzz import fuzz
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, Form, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="ALIBABA SPEAKER ESTIMATER")
templates = Jinja2Templates(directory="templates")

# ==========================
# 1. LOGIC & HELPERS (Refactored from your script)
# ==========================

# Configuration Constants
DEFAULT_PIXELS_PER_METER = 60
OCR_SCALES = [1.0, 1.7, 2.4, 3.0]
CONFIDENCE_THRESHOLD = 0.1
FILL_COLOR = (0, 255, 0)
CONTOUR_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 0)

# Init EasyOCR globally (load once to save memory/time)
reader = easyocr.Reader(["en"], gpu=False)

def normalize_text(text: str) -> str:
    text = text.upper()
    ocr_corrections = {
        "NASHROOM": "WASHROOM", "WASHROON": "WASHROOM", "WASDROOM": "WASHROOM", "W4SHROOM": "WASHROOM",
        "AEWATERPUNP": "NEWWATERPUMPROOM", "FIRSTAIDRMPCLCERM": "POLICEROOM", 
        "FIRST AID RM PCLCERM": "POLICEROOM", "ACCESSIBLE": "ACCESSIBLEWASHROOM",
        "ACCESSIP": "ACCESSIBLEWASHROOM", "ACCESSIPLEWASHROOM": "ACCESSIBLEWASHROOM",
        "GSQ": "GSO", "AHLR": "AHU", "AHRMJ": "AHU", "CHMP R": "CHWP", "CHWPRM": "CHWP RM",
        "YS PLENUM": "VSPLENUM", "( ISF": "CE/ISCS ROOM", "MLS": "MIS",
        "TBYPASSLOBEY": "BYPASSLOBBY", "T BYPASS LOBEY": "BYPASSLOBBY", 
        " TBYPASS LOBEY": "BYPASSLOBBY", "BY-PASS LOBEY": "BYPASSLOBBY",
        "BYPASS LOBEY": "BYPASSLOBBY", "BYPSSLOBBY": "BYPASSLOBBY",
        "SMCKESTOPLOBBY": "SMOKESTOPLOBBY", "OTEFSEFRM": "OTEF/SEFRM",
        "OTEF/SEFRM2": "OTEF/SEFRM", "OTEFSEFROOM": "OTEF/SEFRM",
        "UPASF": "UPASF1", "UPASE": "UPASF1", "UPASF1": "UPASF1",
    }
    for wrong, right in ocr_corrections.items():
        if wrong in text:
            text = text.replace(wrong, right)
    text = re.sub(r"[^A-Z/ ]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text

def remove_black_dotted_lines(img: np.ndarray, dark_max=90, connect_size=3, line_len=35, max_thickness=3) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, dark_max)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (connect_size, connect_size))
    connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    horiz = cv2.morphologyEx(connected, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(connected, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    line_mask = cv2.add(horiz, vert)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    filtered = np.zeros_like(line_mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if (w > line_len and h <= max_thickness) or (h > line_len and w <= max_thickness):
            filtered[labels == i] = 255

    filtered = cv2.dilate(filtered, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(img, filtered, 3, cv2.INPAINT_TELEA)
    return cleaned

def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def get_speaker_type_from_text(text: str, room_map: dict):
    norm_text = normalize_text(text)
    if norm_text in room_map:
        return norm_text, room_map[norm_text]
    for room, spk in room_map.items():
        if room in norm_text:
            return room, spk
    best, best_score = None, 0
    for room in room_map:
        score = fuzz.token_sort_ratio(norm_text, room)
        if score > best_score:
            best_score = score
            best = room
    if best_score >= 70:
        return best, room_map[best]
    return None, "Room not found"

def bbox_center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def draw_label(img_draw, x, y, lines, line_scale=0.6):
    y0 = y
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    sizes = [cv2.getTextSize(t, font, line_scale, thickness)[0] for t in lines]
    if not sizes: return
    box_w = max(sz[0] for sz in sizes) + pad * 2
    box_h = sum(sz[1] for sz in sizes) + pad * (len(lines) + 1)
    
    # Boundary checks to keep label in image
    h_img, w_img = img_draw.shape[:2]
    top_left_y = y0 - box_h
    if top_left_y < 0: top_left_y = 0
    if x + box_w > w_img: x = w_img - box_w

    top_left = (x, top_left_y)
    bot_right = (x + box_w, top_left_y + box_h)

    cv2.rectangle(img_draw, top_left, bot_right, (255, 255, 255), -1)
    y_cursor = top_left_y + pad + sizes[0][1]
    for i, t in enumerate(lines):
        color = (0, 0, 255) if "Spacing" in t else TEXT_COLOR
        cv2.putText(img_draw, t, (x + pad, y_cursor), font, line_scale, color, thickness)
        if i + 1 < len(lines):
            y_cursor += sizes[i + 1][1] + pad

# ==========================
# 2. SESSION MANAGEMENT
# ==========================

class SessionData:
    def __init__(self):
        self.original_img: Optional[np.ndarray] = None
        self.current_img: Optional[np.ndarray] = None # The one with drawings
        self.room_map: dict = {}
        self.ocr_items: list = []
        
        # History Stacks
        self.image_history: list = [] # List of np.ndarray
        self.results_history: list = [] # List of dicts (row data)
        
        # Settings
        self.px_per_m = DEFAULT_PIXELS_PER_METER
        self.grid_mode = "black_dotted"
        self.rules = {"spherical": 6, "horn": 12, "ceiling": 5}

# Simple in-memory store
sessions = {}

# ==========================
# 3. API ROUTES
# ==========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_files(
    image: UploadFile = File(...),
    excel: UploadFile = File(...),
    px_per_m: float = Form(60.0),
    grid_mode: str = Form("black_dotted"),
    spherical_m: float = Form(6.0),
    horn_m: float = Form(12.0),
    ceiling_m: float = Form(5.0)
):
    session_id = str(uuid.uuid4())
    session = SessionData()
    session.px_per_m = px_per_m
    session.grid_mode = grid_mode
    session.rules = {"spherical": spherical_m, "horn": horn_m, "ceiling": ceiling_m}

    # 1. Process Excel
    contents_excel = await excel.read()
    df = pd.read_excel(io.BytesIO(contents_excel))
    mapping = {}
    for col in df.columns:
        speaker_type = col.split()[0]
        for val in df[col].dropna():
            for part in str(val).split("/"):
                norm = normalize_text(part)
                if norm:
                    mapping[norm] = speaker_type
    session.room_map = mapping

    # 2. Process Image
    contents_image = await image.read()
    nparr = np.frombuffer(contents_image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    session.original_img = img.copy()
    session.current_img = img.copy()
    session.image_history = [img.copy()] # Initial state

    # 3. Run OCR (One-time setup)
    # Preprocess
    ocr_input = img.copy()
    if grid_mode == "black_dotted":
        ocr_input = remove_black_dotted_lines(ocr_input)
    ocr_input = enhance_for_ocr(ocr_input)

    # Multiscale OCR
    all_items = []
    for s in OCR_SCALES:
        resized = cv2.resize(ocr_input, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(resized, detail=1, contrast_ths=0.05, text_threshold=0.2, low_text=0.2)
        for (bbox, text, conf) in results:
            if conf < CONFIDENCE_THRESHOLD: continue
            # Map back to original
            bbox_unscaled = [(p[0]/s, p[1]/s) for p in bbox]
            cx, cy = bbox_center(bbox_unscaled)
            norm = normalize_text(text)
            if len(norm) < 3 or len(norm) > 25: continue
            all_items.append({
                "text": text, "norm": norm, "conf": float(conf),
                "cx": cx, "cy": cy
            })
    session.ocr_items = all_items

    sessions[session_id] = session
    
    # Return base64 image
    _, buffer = cv2.imencode('.png', session.current_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"session_id": session_id, "image": f"data:image/png;base64,{img_b64}"}

class ClickRequest(BaseModel):
    session_id: str
    x: int
    y: int

@app.post("/api/click")
async def process_click(req: ClickRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session expired")
    
    sess = sessions[req.session_id]
    img = sess.current_img
    h, w = img.shape[:2]

    # Save state BEFORE drawing for Undo
    sess.image_history.append(img.copy())

    # Flood fill
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Open CV floodfill flags
    flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    
    # We must operate on a mutable copy to find contour, but we draw on sess.current_img
    temp_flood = img.copy()
    cv2.floodFill(temp_flood, mask, (req.x, req.y), FILL_COLOR, loDiff=(5,5,5), upDiff=(5,5,5), flags=flood_flags)
    
    actual_mask = mask[1:-1, 1:-1]
    contours, _ = cv2.findContours(actual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Revert history push if invalid click
        sess.image_history.pop()
        return {"error": "No region found"}

    cnt = max(contours, key=cv2.contourArea)
    
    # Calculations
    perimeter_px = cv2.arcLength(cnt, True)
    perimeter_m = perimeter_px / sess.px_per_m
    
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx_room, cy_room = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cx_room, cy_room = req.x, req.y

    # Find nearest OCR
    best_match = None
    best_d = float("inf")
    for item in sess.ocr_items:
        dx = item["cx"] - cx_room
        dy = item["cy"] - cy_room
        dist = math.hypot(dx, dy)
        if dist < best_d:
            best_d = dist
            best_match = item
    
    room_name = "Unknown"
    speaker_type = "Unknown"
    raw_ocr = ""
    
    # Distance threshold for matching (250px)
    if best_match and best_d < 250:
        raw_ocr = best_match['text']
        room_name, speaker_type = get_speaker_type_from_text(raw_ocr, sess.room_map)
    else:
        room_name = "(No OCR nearby)"

    # Determine spacing
    spacing_m = sess.rules["spherical"]
    rule_name = "SPHERICAL"
    
    if speaker_type:
        st_upper = speaker_type.upper()
        if "HORN" in st_upper:
            spacing_m = sess.rules["horn"]
            rule_name = "HORN"
        elif "CEILING" in st_upper:
            spacing_m = sess.rules["ceiling"]
            rule_name = "CEILING"

    speakers_needed = max(1, int(perimeter_m / spacing_m))

    # Draw result permanently on current image
    cv2.drawContours(sess.current_img, [cnt], -1, CONTOUR_COLOR, 2)
    
    room_idx = len(sess.results_history) + 1
    label1 = f"#{room_idx} Perim: {perimeter_m:.2f}m"
    label2 = f"Speakers ({spacing_m}m): {speakers_needed}"
    label3 = f"Room: {room_name}"
    label4 = f"Rule: {rule_name}"
    
    draw_label(sess.current_img, req.x, req.y, [label1, label2, label3, label4])

    # Save Result Data
    result_data = {
        "room_idx": room_idx,
        "perimeter": round(perimeter_m, 2),
        "room_name": room_name,
        "speaker_type": speaker_type,
        "speakers_needed": speakers_needed,
        "spacing_rule": rule_name
    }
    sess.results_history.append(result_data)

    # Return
    _, buffer = cv2.imencode('.png', sess.current_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image": f"data:image/png;base64,{img_b64}",
        "data": result_data
    }

@app.post("/api/undo")
async def undo_action(req: ClickRequest): # reusing model for session_id
    if req.session_id not in sessions:
        return {"error": "Session not found"}
    sess = sessions[req.session_id]
    
    if len(sess.image_history) > 0:
        # Restore image
        sess.current_img = sess.image_history.pop()
        # Remove last data entry
        if sess.results_history:
            sess.results_history.pop()
            
        _, buffer = cv2.imencode('.png', sess.current_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return {
            "image": f"data:image/png;base64,{img_b64}",
            "history_len": len(sess.results_history)
        }
    return {"status": "nothing to undo"}

@app.get("/api/download/csv")
async def download_csv(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404)
    sess = sessions[session_id]
    df = pd.DataFrame(sess.results_history)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=results.csv"
    return response

@app.get("/api/download/image")
async def download_image(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404)
    sess = sessions[session_id]
    _, buffer = cv2.imencode('.png', sess.current_img)
    return StreamingResponse(io.BytesIO(buffer), media_type="image/png")