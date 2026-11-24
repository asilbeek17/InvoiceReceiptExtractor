"""
Generates 30 fake invoices/receipts into ./samples plus samples/ground_truth.json
(the correct answers, used by evaluate.py to measure accuracy).

Variety: 15 invoices + 15 receipts, different vendors, currencies ($ € £), date formats,
tax rates, item counts, and ~25% deliberately messy "photo-like" images (tilt, blur, noise).
"""
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(7)
np.random.seed(7)
OUT = "samples"

VENDORS = ["Acme Office Supplies Ltd", "Northwind Traders", "Blue Harbor Hardware", "Sunrise Bakery & Cafe",
           "Pixel Print Studio", "Greenfield Organic Market", "Metro Auto Parts", "Lumen Electric Co",
           "Orchid Spa & Wellness", "Summit Cloud Services", "Maple Street Pharmacy", "Urban Bike Works",
           "Golden Wok Kitchen", "Bright Minds Tutoring", "Harbor View Hotel", "Coastal Fresh Seafood",
           "Redwood Furniture Co", "Silver Spoon Diner", "Nova Tech Repairs", "Oakwood Garden Centre",
           "Pine & Co Stationers", "Starlight Cinema", "Quick Fix Plumbing", "Velvet Books & Music",
           "Aqua Pure Laundry", "Iron Gym Fitness", "Daily Grind Coffee", "Prime Print and Copy",
           "Falcon Logistics Ltd", "Willow Pet Supplies"]
PRODUCTS = [("Printer paper A4", 6.5), ("Ink cartridge black", 24.0), ("Stapler heavy duty", 15.99),
            ("Desk lamp LED", 29.9), ("USB-C cable 2m", 12.49), ("Notebook A5 pack", 8.75),
            ("Consulting hour", 85.0), ("Website hosting monthly", 19.0), ("Cappuccino", 4.5),
            ("Blueberry muffin", 3.25), ("Chicken sandwich", 9.75), ("Green salad", 7.8),
            ("Espresso", 3.1), ("Sparkling water", 2.4), ("Phone screen repair", 89.0),
            ("Brake pads set", 54.2), ("Oil filter", 11.6), ("Garden soil 20L", 9.9),
            ("Potted fern", 14.5), ("Cleaning service", 60.0), ("Vitamin C 60 tabs", 10.95),
            ("Organic apples 1kg", 4.2), ("Whole milk 2L", 2.7), ("Sourdough loaf", 5.5),
            ("Delivery fee", 6.0), ("Installation labour", 120.0), ("Memory card 64GB", 18.99)]
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
DATE_FORMATS = ["%B %d, %Y", "%Y-%m-%d", "%d %b %Y"]
PAYMENTS = ["VISA ending 4242", "MASTERCARD ending 1881", "CASH", "PAYPAL"]


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def messy(img):
    """Make it look like a phone photo: grey background, tilt, blur, noise, low contrast."""
    angle = random.uniform(-2.5, 2.5)
    img = img.rotate(angle, expand=True, fillcolor=(205, 205, 205), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.1)))
    arr = np.asarray(img).astype(np.float32)
    arr = arr * random.uniform(0.82, 0.92) + 12 + np.random.normal(0, 7, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def money(v, sym):
    return f"{sym}{v:,.2f}"


def make_doc(i, kind):
    vendor = VENDORS[i % len(VENDORS)]
    sym = random.choices(list(SYMBOLS), weights=[6, 2, 2])[0]
    fmt = random.choice(DATE_FORMATS)
    date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 900))
    rate = random.choice([0, 5, 7, 8, 10, 20])
    n_items = random.randint(2, 6) if kind == "invoice" else random.randint(2, 5)
    items = []
    for desc, price in random.sample(PRODUCTS, n_items):
        qty = random.randint(1, 6) if kind == "invoice" else 1
        items.append({"description": desc, "quantity": qty, "unit_price": price, "amount": round(qty * price, 2)})
    subtotal = round(sum(it["amount"] for it in items), 2)
    tax = round(subtotal * rate / 100, 2)
    total = round(subtotal + tax, 2)
    number = (f"INV-{date.year}-{random.randint(1000, 9999)}" if kind == "invoice" else str(random.randint(10000, 99999)))
    truth = {"document_type": kind, "vendor_name": vendor, "invoice_number": number,
             "date": date.strftime("%Y-%m-%d"), "currency": SYMBOLS[sym], "subtotal": subtotal,
             "tax": tax, "total": total, "n_items": n_items}

    width = 900 if kind == "invoice" else 640
    height = 520 + 46 * n_items
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    f, fb = font(26), font(30)
    right = width - 40
    y = 30
    d.text((40, y), vendor if kind == "invoice" else vendor.upper(), fill="black", font=fb); y += 52
    if kind == "invoice":
        d.text((40, y), "INVOICE", fill="black", font=f); y += 40
        d.text((40, y), f"Invoice No: {number}", fill="black", font=f); y += 38
        d.text((40, y), f"Date: {date.strftime(fmt)}", fill="black", font=f); y += 38
        due = (date + timedelta(days=30)).strftime(fmt)
        d.text((40, y), f"Due Date: {due}", fill="black", font=f); y += 52
        for it in items:
            d.text((40, y), it["description"], fill="black", font=f)
            d.text((500, y), str(it["quantity"]), fill="black", font=f)
            d.text((680, y), money(it["unit_price"], sym), fill="black", font=f, anchor="ra")
            d.text((right, y), money(it["amount"], sym), fill="black", font=f, anchor="ra")
            y += 42
        y += 14
        d.text((40, y), "Subtotal", fill="black", font=f); d.text((right, y), money(subtotal, sym), fill="black", font=f, anchor="ra"); y += 40
        if rate:
            d.text((40, y), f"Tax ({rate}%)", fill="black", font=f); d.text((right, y), money(tax, sym), fill="black", font=f, anchor="ra"); y += 40
        else:
            truth["tax"] = None
        d.text((40, y), "Total Due", fill="black", font=fb); d.text((right, y), money(total, sym), fill="black", font=fb, anchor="ra")
    else:
        d.text((40, y), f"Receipt #{number}", fill="black", font=f); y += 38
        d.text((40, y), date.strftime(fmt), fill="black", font=f); y += 52
        for it in items:
            d.text((40, y), it["description"], fill="black", font=f)
            d.text((right, y), f"{it['amount']:.2f}", fill="black", font=f, anchor="ra")
            y += 42
        y += 14
        d.text((40, y), "Subtotal", fill="black", font=f); d.text((right, y), f"{subtotal:.2f}", fill="black", font=f, anchor="ra"); y += 40
        if rate:
            d.text((40, y), "Tax", fill="black", font=f); d.text((right, y), f"{tax:.2f}", fill="black", font=f, anchor="ra"); y += 40
        else:
            truth["tax"] = None
        d.text((40, y), "TOTAL", fill="black", font=fb); d.text((right, y), money(total, sym), fill="black", font=fb, anchor="ra"); y += 52
        d.text((40, y), random.choice(PAYMENTS), fill="black", font=f)
    img = img.crop((0, 0, width, min(height, y + 60)))

    hard = i % 4 == 3
    if hard:
        img = messy(img)
    truth["difficulty"] = "hard" if hard else "clean"
    return img, truth


os.makedirs(OUT, exist_ok=True)
ground_truth = {}
for i in range(30):
    kind = "invoice" if i % 2 == 0 else "receipt"
    img, truth = make_doc(i, kind)
    name = f"sample_{i + 1:02d}_{kind}.png"
    img.save(os.path.join(OUT, name))
    ground_truth[name] = truth
with open(os.path.join(OUT, "ground_truth.json"), "w") as fh:
    json.dump(ground_truth, fh, indent=2)
print(f"Created {len(ground_truth)} sample documents in ./{OUT} (+ ground_truth.json)")
