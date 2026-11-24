import io
import json
import os
import time

import pandas as pd
import streamlit as st

from extractor import DEFAULT_MODEL, extract_document, line_item_rows, summary_row

st.set_page_config(page_title="Invoice & Receipt Extractor", page_icon="🧾", layout="wide")

SAMPLE_DIR = "samples"

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.hero {padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);}
.hero h1 {color: #fff; margin: 0; font-size: 2rem; padding: 0;}
.hero p {color: rgba(255,255,255,.9); margin: .4rem 0 0 0; font-size: 1.02rem;}
.steps {display: flex; gap: .75rem; margin: .25rem 0 1.2rem 0; flex-wrap: wrap;}
.step {flex: 1; min-width: 190px; padding: .8rem 1rem; border-radius: 12px;
       border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);}
.step b {display: block; margin-bottom: .1rem;}
.step span {font-size: .88rem; opacity: .75;}
.badge {display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-size: .8rem;
        font-weight: 600; margin-left: .5rem; vertical-align: middle;}
.badge-ok {background: rgba(34,197,94,.18); color: #16a34a;}
.badge-warn {background: rgba(245,158,11,.2); color: #d97706;}
.kv {display: grid; grid-template-columns: 1fr 1fr; gap: .6rem 1rem; margin: .3rem 0 .8rem 0;}
.kv small {display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; opacity: .6;}
.kv span {font-weight: 600; font-size: 1rem;}
</style>
<div class="hero">
  <h1>🧾 Invoice &amp; Receipt Extractor</h1>
  <p>Turn photos, scans and PDFs of invoices into clean, checked, export-ready data in seconds.</p>
</div>
<div class="steps">
  <div class="step"><b>1 · Upload</b><span>Images or PDFs, one or many</span></div>
  <div class="step"><b>2 · Extract</b><span>Vendor, dates, totals, line items</span></div>
  <div class="step"><b>3 · Verify</b><span>Math is checked, errors are flagged</span></div>
  <div class="step"><b>4 · Export</b><span>Excel, CSV or JSON</span></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.success("Runs 100% on this computer. Documents never leave your machine, and there is no per-document cost.",
               icon="🔒")
    manual_minutes = st.number_input("Manual entry time per document (min)", 0.5, 30.0, 3.0, 0.5,
                                     help="Your own estimate. Used only for the 'time saved' figure.")

    # Optional, off by default: an AI engine for harder documents (requires your own API key)
    engine, api_key, model = "local", None, DEFAULT_MODEL
    with st.expander("Advanced (optional)"):
        if st.checkbox("Use AI engine instead (needs an Anthropic API key)"):
            engine = "ai"
            try:
                default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                default_key = ""
            api_key = st.text_input("Anthropic API key", value=default_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                                    type="password")
            model = st.text_input("Model", value=DEFAULT_MODEL)

# ---------------------------------------------------------------- input
files = st.file_uploader("Drop invoices or receipts here", type=["png", "jpg", "jpeg", "webp", "pdf"],
                         accept_multiple_files=True)
col_a, col_b, _ = st.columns([1, 1.3, 3])
run_clicked = col_a.button("✨ Extract data", type="primary", disabled=not files, width="stretch")
sample_clicked = col_b.button("📎 Try sample documents", width="stretch",
                              disabled=not os.path.isdir(SAMPLE_DIR),
                              help="Runs 6 of the 30 fake documents in the samples folder (a mix of invoices, "
                                   "receipts and messy photos). Upload all 30 yourself to test a big batch.")

docs = None
if sample_clicked:
    names = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")))
    docs = [(f, open(os.path.join(SAMPLE_DIR, f), "rb").read()) for f in names[::5][:6]]
elif run_clicked:
    docs = [(f.name, f.getvalue()) for f in files]

if docs:
    results, bar = [], st.progress(0.0, text="Starting...")
    t0 = time.perf_counter()
    for n, (name, data) in enumerate(docs, 1):
        bar.progress((n - 1) / len(docs), text=f"Reading {name} ({n}/{len(docs)})")
        try:
            res = extract_document(data, name, engine=engine, api_key=api_key, model=model)
        except Exception as e:  # keep going if one file fails
            res = {"filename": name, "engine": engine, "data": {"line_items": []},
                   "warnings": [f"Failed: {e}"]}
        res["_bytes"] = data
        results.append(res)
    bar.empty()
    st.session_state["results"] = results
    st.session_state["elapsed"] = time.perf_counter() - t0

results = st.session_state.get("results")


def money(v, cur):
    return "—" if v is None else f"{cur + ' ' if cur else ''}{v:,.2f}"


# ---------------------------------------------------------------- results
if results:
    summary = pd.DataFrame([summary_row(r) for r in results])
    summary.insert(0, "status", ["✅ OK" if not r["warnings"] else "⚠️ Review" for r in results])
    items = pd.DataFrame([row for r in results for row in line_item_rows(r)])
    flagged = int((summary["status"] != "✅ OK").sum())
    saved = max(0.0, len(results) * manual_minutes - st.session_state.get("elapsed", 0) / 60)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents processed", len(results), border=True)
    m2.metric("Total amount", f"{pd.to_numeric(summary['total'], errors='coerce').sum():,.2f}", border=True)
    m3.metric("Need review", flagged, border=True,
              help="Totals that don't add up, or missing key fields")
    m4.metric("Est. time saved", f"{saved:.0f} min", border=True,
              help=f"Assumes {manual_minutes:g} min of manual entry per document (sidebar)")

    st.subheader("Documents")
    for r in results:
        d, ok = r["data"], not r["warnings"]
        cur = d.get("currency") or ""
        with st.container(border=True):
            badge = ('<span class="badge badge-ok">Totals verified</span>' if ok
                     else '<span class="badge badge-warn">Needs review</span>')
            st.markdown(f"**{r['filename']}** {badge}", unsafe_allow_html=True)
            left, right = st.columns([1, 1.4])
            with left:
                if r["filename"].lower().endswith("pdf"):
                    st.caption("PDF preview not shown")
                else:
                    st.image(r["_bytes"], width="stretch")
            with right:
                st.markdown(f"""
<div class="kv">
  <div><small>Vendor</small><span>{d.get('vendor_name') or '—'}</span></div>
  <div><small>Invoice / receipt no.</small><span>{d.get('invoice_number') or '—'}</span></div>
  <div><small>Date</small><span>{d.get('date') or '—'}</span></div>
  <div><small>Due date</small><span>{d.get('due_date') or '—'}</span></div>
  <div><small>Subtotal</small><span>{money(d.get('subtotal'), cur)}</span></div>
  <div><small>Tax</small><span>{money(d.get('tax'), cur)}</span></div>
  <div><small>Total</small><span>{money(d.get('total'), cur)}</span></div>
  <div><small>Payment</small><span>{d.get('payment_method') or '—'}</span></div>
</div>""", unsafe_allow_html=True)
                for w in r["warnings"]:
                    st.warning(w, icon="⚠️")
                li = d.get("line_items") or []
                if li:
                    st.dataframe(pd.DataFrame(li), width="stretch", hide_index=True, column_config={
                        "description": "Description", "quantity": "Qty",
                        "unit_price": st.column_config.NumberColumn("Unit price", format="%.2f"),
                        "amount": st.column_config.NumberColumn("Amount", format="%.2f")})
                if d.get("_raw_text"):
                    with st.popover("Raw OCR text"):
                        st.text(d["_raw_text"])

    st.subheader("Review & export")
    st.caption("Fix any cell below before exporting. Your edits are included in the files.")
    edited = st.data_editor(summary, width="stretch", hide_index=True, num_rows="fixed",
                            disabled=["status", "file"],
                            column_config={"needs_review": st.column_config.TextColumn("issues", width="large")})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        edited.to_excel(xw, sheet_name="Documents", index=False)
        (items if not items.empty else pd.DataFrame(columns=["file"])).to_excel(
            xw, sheet_name="Line items", index=False)
    json_out = json.dumps(
        [{"file": r["filename"], "warnings": r["warnings"],
          **{k: v for k, v in r["data"].items() if k != "_raw_text"}} for r in results],
        indent=2, ensure_ascii=False)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Excel (.xlsx)", buf.getvalue(), "extracted.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", width="stretch")
    d2.download_button("⬇️ CSV (summary)", edited.to_csv(index=False), "extracted.csv", "text/csv",
                       width="stretch")
    d3.download_button("⬇️ JSON (full)", json_out, "extracted.json", "application/json", width="stretch")
else:
    st.info("Upload documents above, or click **Try sample documents** to see how it works.", icon="👆")
