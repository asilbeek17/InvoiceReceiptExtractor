"""
Core extraction logic for invoices and receipts.

Two engines:
  "ai"    : Claude vision with a forced JSON schema (most accurate, needs an API key)
  "local" : Tesseract OCR + regex rules (free, offline, rougher)

Both return the same structure, then `validate()` flags suspicious results
(e.g. line items that don't add up to the total).
"""
import base64
import io
import os
import re

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}

# ----------------------------------------------------------------------------
# Output schema (also used as the tool schema for Claude)
# ----------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": ["invoice", "receipt", "other"]},
        "vendor_name": {"type": ["string", "null"]},
        "vendor_address": {"type": ["string", "null"]},
        "invoice_number": {"type": ["string", "null"], "description": "Invoice or receipt number"},
        "date": {"type": ["string", "null"], "description": "Issue date, YYYY-MM-DD"},
        "due_date": {"type": ["string", "null"], "description": "Payment due date, YYYY-MM-DD"},
        "currency": {"type": ["string", "null"], "description": "ISO code like USD, EUR, KRW"},
        "subtotal": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
        "payment_method": {"type": ["string", "null"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": ["number", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["description"],
            },
        },
    },
    "required": ["document_type", "line_items"],
}

PROMPT = (
    "Extract the data from this invoice or receipt. Rules: use null for anything not "
    "present (never guess); dates as YYYY-MM-DD; numbers as plain numbers without "
    "currency symbols or thousands separators; currency as an ISO code; include every "
    "line item exactly as written. Call the record_document tool with the result."
)

EMPTY = {
    "document_type": "other", "vendor_name": None, "vendor_address": None,
    "invoice_number": None, "date": None, "due_date": None, "currency": None,
    "subtotal": None, "tax": None, "total": None, "payment_method": None, "line_items": [],
}


def _ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ----------------------------------------------------------------------------
# Engine 1: Claude vision
# ----------------------------------------------------------------------------
def _shrink_image(data, max_side=1800):
    """Downscale big photos (cheaper and faster, no accuracy loss for documents)."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    img.thumbnail((max_side, max_side))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def extract_with_claude(file_bytes, filename, api_key=None, model=None, client=None):
    ext = _ext(filename)
    if client is None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No API key. Add one in the sidebar or set ANTHROPIC_API_KEY.")
        client = anthropic.Anthropic(api_key=key)

    if ext == "pdf":
        block = {"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64.b64encode(file_bytes).decode()}}
    elif ext in IMAGE_EXTS:
        data, media = _shrink_image(file_bytes)
        block = {"type": "image", "source": {
            "type": "base64", "media_type": media, "data": base64.b64encode(data).decode()}}
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=2000,
        tools=[{"name": "record_document",
                "description": "Record the structured data extracted from the document.",
                "input_schema": SCHEMA}],
        tool_choice={"type": "tool", "name": "record_document"},
        messages=[{"role": "user", "content": [block, {"type": "text", "text": PROMPT}]}],
    )
    for item in resp.content:
        if item.type == "tool_use":
            return {**EMPTY, **item.input}
    raise RuntimeError("The model returned no structured result.")


# ----------------------------------------------------------------------------
# Engine 2: Tesseract OCR + rules
# ----------------------------------------------------------------------------
MONEY = r"-?\d[\d,]*\.\d{2}"
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "₩": "KRW", "¥": "JPY"}
SKIP_WORDS = ("subtotal", "sub total", "total", "tax", "vat", "gst", "balance", "amount due",
              "change", "cash", "visa", "mastercard", "discount", "tip", "shipping")


def _load_gray(data):
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    if min(img.size) < 1000:  # small photos OCR better when upscaled
        scale = 1000 / min(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


def ocr_variants(data):
    """Same image, three preprocessing styles. Each is OCR'd; the best result is chosen later."""
    try:
        import pytesseract
        from PIL import ImageFilter, ImageOps
    except ImportError as e:
        raise RuntimeError("Install pytesseract and Tesseract (brew install tesseract).") from e
    gray = _load_gray(data)
    plain = ImageOps.autocontrast(gray)
    # denoise, then force pure black/white (helps blurry, low-contrast photos)
    clean = ImageOps.autocontrast(gray.filter(ImageFilter.MedianFilter(3)), cutoff=2)
    cut = int(sum(clean.getdata()) / (clean.width * clean.height) * 0.78)
    binary = clean.point(lambda p: 255 if p > cut else 0)
    sharp = ImageOps.autocontrast(gray.filter(ImageFilter.SHARPEN))
    return [pytesseract.image_to_string(plain, config="--psm 6"),
            pytesseract.image_to_string(binary, config="--psm 6"),
            pytesseract.image_to_string(sharp, config="--psm 4")]


def ocr_image(data):
    return ocr_variants(data)[0]


def pdf_text(data):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _money(s):
    return float(s.replace(",", ""))


def _find_amount(lines, keywords, exclude=()):
    """Amount on the last line containing one of the keywords."""
    found = None
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in keywords) and not any(x in low for x in exclude):
            nums = re.findall(MONEY, ln)
            if nums:
                found = _money(nums[-1])
    return found


def _parse_date(text):
    from dateutil import parser as dparser

    patterns = [
        r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b",
        r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{4}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return dparser.parse(m.group(0)).strftime("%Y-%m-%d")
            except (ValueError, OverflowError):
                continue
    return None


def extract_with_rules(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {**EMPTY, "line_items": []}
    if not lines:
        return out

    low_text = text.lower()
    out["document_type"] = "invoice" if "invoice" in low_text else (
        "receipt" if "receipt" in low_text else "other")

    for ln in lines[:6]:  # vendor = first line that is mostly letters and not a title word
        if len(re.findall(r"[A-Za-z]", ln)) >= 3 and ln.lower() not in ("invoice", "receipt", "tax invoice"):
            out["vendor_name"] = ln
            break

    # label followed (on the same line) by a token that contains at least one digit
    for m in re.finditer(r"(?:invoice|inv|receipt|bill|order)[ \t]*(?:no|number|num|#)?[ \t]*[.:#]*[ \t]*"
                         r"([A-Z0-9][A-Z0-9\-/]*\d[A-Z0-9\-/]*)", text, re.I):
        out["invoice_number"] = m.group(1)
        break

    out["date"] = _parse_date(text)
    due_line = next((ln for ln in lines if "due" in ln.lower() and "amount due" not in ln.lower()
                     and "balance" not in ln.lower()), None)
    if due_line:
        out["due_date"] = _parse_date(due_line)

    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            out["currency"] = code
            break
    else:
        m = re.search(r"\b(USD|EUR|GBP|KRW|INR|JPY|CAD|AUD)\b", text)
        out["currency"] = m.group(1) if m else None

    out["subtotal"] = _find_amount(lines, ("subtotal", "sub total", "sub-total"))
    out["tax"] = _find_amount(lines, ("tax", "vat", "gst"), exclude=("subtotal",))
    out["total"] = _find_amount(lines, ("grand total", "total due", "amount due", "balance due", "total"),
                                exclude=("subtotal", "sub total", "sub-total"))

    for pm in ("visa", "mastercard", "amex", "cash", "paypal", "debit", "credit card"):
        if pm in low_text:
            out["payment_method"] = pm
            break

    item_re = re.compile(rf"^(?P<desc>.*?[A-Za-z].*?)\s+(?:(?P<qty>\d+(?:\.\d+)?)\s*[x@]?\s+)?"
                         rf"[^\d\s-]?\s?(?:(?P<unit>{MONEY})\s+)?[^\d\s-]?\s?(?P<amt>{MONEY})$")
    for ln in lines:
        if any(w in ln.lower() for w in SKIP_WORDS):
            continue
        m = item_re.match(ln)
        if m:
            out["line_items"].append({
                "description": m.group("desc").strip(" .:-"),
                "quantity": float(m.group("qty")) if m.group("qty") else None,
                "unit_price": _money(m.group("unit")) if m.group("unit") else None,
                "amount": _money(m.group("amt")),
            })
    return out


def _quality(data):
    """Higher = more complete and self-consistent extraction."""
    key = ["vendor_name", "date", "subtotal", "total"]
    filled = sum(data.get(k) is not None for k in key) + (1 if data.get("line_items") else 0)
    return filled - 2 * len(validate(data))


def extract_with_local(file_bytes, filename):
    ext = _ext(filename)
    if ext == "pdf":
        text = pdf_text(file_bytes)
        if not text.strip():
            raise ValueError("This PDF is a scan (no text layer). Upload it as an image (PNG/JPG) instead.")
        texts = [text]
    elif ext in IMAGE_EXTS:
        texts = ocr_variants(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    # parse every OCR variant and keep the one that is most complete and passes the math checks
    best_data, best_text, best_score = None, "", -99
    for text in texts:
        data = extract_with_rules(text)
        score = _quality(data)
        if score > best_score:
            best_data, best_text, best_score = data, text, score
    best_data["_raw_text"] = best_text
    return best_data


# ----------------------------------------------------------------------------
# Validation: flag results a human should double-check
# ----------------------------------------------------------------------------
def validate(data):
    warnings = []
    total, subtotal, tax = data.get("total"), data.get("subtotal"), data.get("tax")
    items = data.get("line_items") or []

    if total is None:
        warnings.append("Total not found")
    if not data.get("date"):
        warnings.append("Date not found")
    if not data.get("vendor_name"):
        warnings.append("Vendor not found")

    amounts = [i["amount"] for i in items if i.get("amount") is not None]
    if amounts:
        items_sum = round(sum(amounts), 2)
        ref = subtotal if subtotal is not None else total
        if ref is not None and abs(items_sum - ref) > max(0.02, 0.01 * abs(ref)):
            label = "subtotal" if subtotal is not None else "total"
            warnings.append(f"Line items add up to {items_sum:,.2f} but {label} is {ref:,.2f}")
    elif not items:
        warnings.append("No line items found")

    if subtotal is not None and tax is not None and total is not None:
        if abs(subtotal + tax - total) > max(0.02, 0.01 * abs(total)):
            warnings.append(f"Subtotal + tax = {subtotal + tax:,.2f} but total is {total:,.2f}")

    if subtotal is not None and total is not None and tax is None and total - subtotal > 0.02:
        warnings.append(f"Total is {total - subtotal:,.2f} higher than subtotal but no tax was found")

    for i in items:
        q, u, a = i.get("quantity"), i.get("unit_price"), i.get("amount")
        if None not in (q, u, a) and abs(q * u - a) > max(0.02, 0.01 * abs(a)):
            warnings.append(f"'{i['description'][:30]}': {q} x {u} is not {a}")
    return warnings


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def extract_document(file_bytes, filename, engine="local", api_key=None, model=None, client=None):
    if engine == "ai":
        data = extract_with_claude(file_bytes, filename, api_key, model, client)
    else:
        data = extract_with_local(file_bytes, filename)
    return {"filename": filename, "engine": engine, "data": data, "warnings": validate(data)}


def summary_row(result):
    d = result["data"]
    return {
        "file": result["filename"], "type": d.get("document_type"), "vendor": d.get("vendor_name"),
        "invoice_number": d.get("invoice_number"), "date": d.get("date"), "due_date": d.get("due_date"),
        "currency": d.get("currency"), "subtotal": d.get("subtotal"), "tax": d.get("tax"),
        "total": d.get("total"), "needs_review": "; ".join(result["warnings"]),
    }


def line_item_rows(result):
    d = result["data"]
    return [{"file": result["filename"], "vendor": d.get("vendor_name"), **i}
            for i in (d.get("line_items") or [])]
