"""
NeuroSymbolic Verifier v3 -- Streamlit App
Claude claude-sonnet-4-5 · sentence-transformers · Qdrant · Rewrite Loop · Insight Panel
"""

import streamlit as st
import sys, os, json, time, uuid, difflib

from concurrent.futures import ThreadPoolExecutor
import io, re as _re


def _draft_to_pdf(draft_text: str, run_id: str = "", ltn_score: float = None,
                  rules_passed: int = 0, rules_total: int = 0,
                  iterations_used: int = 1, llm_label: str = "Claude") -> bytes:
    """Convert verified draft markdown text to a clean PDF. Returns raw bytes."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 not installed. Run: pip install fpdf2")

    class PDF(FPDF):
        def header(self):
            _hcw = self.w - self.l_margin - self.r_margin
            # ── Line 1: title (left, bold, 9pt) ──────────────────────────────
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(100, 100, 100)
            self.set_x(self.l_margin)
            self.cell(_hcw, 5,
                      "NeuroSymbolic Verifier  --  Verified Draft Output",
                      align="L", ln=True)
            # ── Line 2: badge metadata (right, regular, 7.5pt) ───────────────
            # Putting the badge on its own line at smaller font guarantees it
            # always fits regardless of how long the model/provider name is.
            if ltn_score is not None:
                badge = (f"LTN {ltn_score:.4f}  |  {rules_passed}/{rules_total} rules"
                         f"  |  {iterations_used} iter(s)"
                         f"  |  {llm_label}  |  run {run_id}")
                self.set_font("Helvetica", "", 7)
                self.set_text_color(150, 150, 150)
                self.set_x(self.l_margin)
                self.cell(_hcw, 4, badge, align="R", ln=True)
            self.ln(1)
            self.set_draw_color(200, 169, 110)
            self.set_line_width(0.4)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(160, 160, 160)
            self.cell(0, 8, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_margins(left=20, top=22, right=20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    C_BODY = (30,  30,  30)
    C_H1   = (27,  58,  92)
    C_H2   = (46, 117, 182)
    C_H3   = (31, 107, 117)
    C_CODE = (60, 100,  60)
    C_GOLD = (170, 130, 60)

    def set_c(c): pdf.set_text_color(*c)

    cw = pdf.w - pdf.l_margin - pdf.r_margin

    import re as _r

    def strip_md(t):
        t = _r.sub(r'\*\*(.*?)\*\*', r'\1', t)
        t = _r.sub(r'\*(.*?)\*',     r'\1', t)
        t = _r.sub(r'`(.*?)`',       r'\1', t)
        return t

    # Helvetica (built-in fpdf2 font) only covers Latin-1 (ISO-8859-1).
    # Sanitize text for fpdf Helvetica (Latin-1 only).
    # Use chr() to avoid any raw/escape string ambiguity.
    _CHAR_MAP = [
        (chr(0x2014), '--'),   # em dash
        (chr(0x2013), '-'),    # en dash
        (chr(0x2018), "'"),   # left single quote
        (chr(0x2019), "'"),   # right single quote
        (chr(0x201C), '"'),   # left double quote
        (chr(0x201D), '"'),   # right double quote
        (chr(0x2022), '-'),    # bullet
        (chr(0x00B0), ' deg'), # degree
        (chr(0x00D7), 'x'),    # multiplication
        (chr(0x2264), '<='),   # less-equal
        (chr(0x2265), '>='),   # greater-equal
        (chr(0x00B1), '+/-'),  # plus-minus
        (chr(0x2026), '...'),  # ellipsis
        (chr(0x00E9), 'e'),    # e acute
        (chr(0x00E8), 'e'),    # e grave
        (chr(0x00EA), 'e'),    # e circumflex
        (chr(0x00E0), 'a'),    # a grave
        (chr(0x00E2), 'a'),    # a circumflex
    ]

    def sanitize(t):
        for uc, asc in _CHAR_MAP:
            t = t.replace(uc, asc)
        # Catch any remaining non-Latin-1 chars with '?'
        return t.encode('latin-1', errors='replace').decode('latin-1')

    def safe(t):
        return sanitize(strip_md(t))

    # Pre-sanitize the entire draft before line-by-line processing
    # This catches em dashes and other Unicode that appear anywhere,
    # including in positions where the line-level safe() call might be skipped.
    draft_text = sanitize(draft_text)

    # ── LaTeX → plain-text converter ──────────────────────────────────────────
    # fpdf2 / Helvetica has zero LaTeX support.  Convert the most common
    # math constructs to readable ASCII so nothing prints as raw backslash junk.
    def _latex_to_plain(s):
        s = s.strip()
        # Strip outer delimiters  \[ \]  \( \)
        if s.startswith(r'\['):  s = s[2:]
        if s.endswith(r'\]'):    s = s[:-2]
        if s.startswith(r'\('):  s = s[2:]
        if s.endswith(r'\)'):    s = s[:-2]
        # \frac{a}{b}  →  (a)/(b)   (repeat for nested fractions)
        import re as _re2
        for _ in range(6):
            s, n = _re2.subn(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
            if not n:
                break
        # \sum_{low}^{high}
        s = _re2.sub(r'\\sum_\{([^}]*)\}\^\{([^}]*)\}', r'sum(\1 to \2)', s)
        s = _re2.sub(r'\\sum_([a-zA-Z0-9=])\^([a-zA-Z0-9])', r'sum(\1 to \2)', s)
        s = _re2.sub(r'\\sum', 'sum', s)
        # \sqrt{x}  →  sqrt(x)
        s = _re2.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
        # Spacing / operator commands
        s = s.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\!', '')
        s = s.replace(r'\cdot', '*').replace(r'\times', 'x').replace(r'\div', '/')
        s = s.replace(r'\pm', '+/-').replace(r'\mp', '-/+')
        s = s.replace(r'\leq', '<=').replace(r'\geq', '>=')
        s = s.replace(r'\le',  '<=').replace(r'\ge',  '>=')
        s = s.replace(r'\neq', '!=').replace(r'\approx', '~=')
        s = s.replace(r'\infty', 'inf').replace(r'\ldots', '...').replace(r'\cdots', '...')
        # Greek letters — common ones
        _greek = {
            r'\theta': 'theta', r'\alpha': 'alpha', r'\beta': 'beta',
            r'\gamma': 'gamma', r'\delta': 'delta', r'\epsilon': 'eps',
            r'\zeta': 'zeta',  r'\eta': 'eta',     r'\kappa': 'kappa',
            r'\lambda': 'lambda',r'\mu': 'mu',      r'\nu': 'nu',
            r'\xi': 'xi',      r'\pi': 'pi',        r'\rho': 'rho',
            r'\sigma': 'sigma', r'\tau': 'tau',     r'\phi': 'phi',
            r'\chi': 'chi',    r'\psi': 'psi',      r'\omega': 'omega',
            r'\Theta': 'Theta', r'\Gamma': 'Gamma', r'\Delta': 'Delta',
            r'\Sigma': 'Sigma', r'\Omega': 'Omega', r'\Lambda': 'Lambda',
        }
        for latex_g, plain_g in _greek.items():
            s = s.replace(latex_g, plain_g)
        # Superscripts  ^{...}  and subscripts  _{...}
        s = _re2.sub(r'\^\{([^}]*)\}', r'^(\1)', s)
        s = _re2.sub(r'_\{([^}]*)\}',  r'_(\1)', s)
        s = _re2.sub(r'\^([a-zA-Z0-9])', r'^\1', s)
        s = _re2.sub(r'_([a-zA-Z0-9])',  r'_\1', s)
        # Formatting commands that should just be removed
        s = _re2.sub(r'\\(?:text|mathrm|mathbf|mathit|mathtt|boldsymbol)\{([^}]*)\}', r'\1', s)
        # Remove remaining unknown \commands
        s = _re2.sub(r'\\[a-zA-Z]+\*?', '', s)
        # Strip remaining bare braces
        s = s.replace('{', '').replace('}', '')
        # Collapse whitespace
        s = _re2.sub(r' {2,}', ' ', s).strip()
        return s

    # Pre-process inline LaTeX  \(...\)  and  $...$  in the full draft text
    # so that any math embedded inside bullets, body text, or headings
    # comes out as readable plain text before line-by-line rendering.
    def _clean_inline_math(text):
        # \( ... \)  inline display
        text = _r.sub(r'\\\((.+?)\\\)', lambda m: _latex_to_plain(m.group(1)), text,
                      flags=_r.DOTALL)
        # $...$  (single dollar, not $$)
        text = _r.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
                      lambda m: _latex_to_plain(m.group(1)), text)
        return text

    draft_text = _clean_inline_math(draft_text)

    # Pre-collect lines so we can look ahead for table blocks
    _lines = draft_text.split("\n")
    _li = 0
    while _li < len(_lines):
        raw_line = _lines[_li]; _li += 1
        line = raw_line.rstrip()

        if not line.strip():
            pdf.ln(3); continue

        # ── Fenced code block: ```lang ... ``` ─────────────────────────────
        # Collect all lines between the opening and closing ``` fence.
        # Render in Courier on a light green background with a fine border.
        if line.strip().startswith("```"):
            code_lines = []
            while _li < len(_lines):
                cl = _lines[_li]; _li += 1
                if cl.strip().startswith("```"):
                    break
                code_lines.append(cl)
            if code_lines:
                pdf.ln(2)
                # Draw a filled rect behind the whole block
                _bx  = pdf.l_margin
                _by  = pdf.get_y()
                _bw  = cw
                _bh  = len(code_lines) * 4.5 + 4
                if _by + _bh > pdf.h - 18:
                    pdf.add_page(); _by = pdf.get_y()
                pdf.set_fill_color(242, 248, 242)
                pdf.set_draw_color(170, 200, 170)
                pdf.set_line_width(0.25)
                pdf.rect(_bx, _by, _bw, _bh, style='FD')
                pdf.set_y(_by + 2)
                for cl in code_lines:
                    pdf.set_font("Courier", "", 8.5); set_c(C_CODE)
                    pdf.set_x(_bx + 3)
                    pdf.multi_cell(_bw - 6, 4.5, sanitize(cl), align="L")
                pdf.set_y(_by + _bh)
                pdf.ln(2)
            continue

        # ── Display math block: \[ ... \] ──────────────────────────────────
        # fpdf2 cannot render LaTeX.  Detect \[...\] blocks (which may span
        # multiple lines), convert to readable plain-text via _latex_to_plain,
        # and render centred in Courier on a pale-blue tinted background.
        _stripped = line.strip()
        if _stripped.startswith(r'\['):
            math_lines = []
            rest = _stripped[2:].strip()
            if r'\]' in rest:
                # Entire block on one line: \[ ... \]
                math_lines.append(rest[:rest.index(r'\]')].strip())
            else:
                if rest:
                    math_lines.append(rest)
                while _li < len(_lines):
                    ml = _lines[_li]; _li += 1
                    if r'\]' in ml:
                        before = ml[:ml.index(r'\]')].strip()
                        if before:
                            math_lines.append(before)
                        break
                    math_lines.append(ml.strip())
            plain = _latex_to_plain(' '.join(math_lines))
            if plain:
                pdf.ln(2)
                pdf.set_font("Courier", "I", 9); set_c((50, 80, 130))
                pdf.set_fill_color(243, 246, 253)
                pdf.set_draw_color(180, 195, 225)
                pdf.set_line_width(0.2)
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(cw - 8, 5.5, sanitize(plain),
                               align="C", fill=True, border=1)
                pdf.ln(2)
            continue
        if line.startswith("|"):
            _tbl_lines = [line]
            while _li < len(_lines) and _lines[_li].strip().startswith("|"):
                _tbl_lines.append(_lines[_li].strip()); _li += 1
            # Filter out separator rows (|---|---| style)
            _data_rows = [r for r in _tbl_lines
                          if not _r.match(r'^[|][-:\s|]+[|]', r)]
            if _data_rows:
                _cells = [[safe(c.strip()) for c in row.strip("|").split("|")]
                          for row in _data_rows]
                _ncols = max(len(row) for row in _cells)
                _col_w = cw / _ncols
                for _ri, _row in enumerate(_cells):
                    _is_hdr = (_ri == 0)
                    pdf.set_font("Helvetica","B" if _is_hdr else "",8)
                    pdf.set_fill_color(230,240,250) if _is_hdr else pdf.set_fill_color(255,255,255)
                    set_c((27,58,92) if _is_hdr else C_BODY)
                    _row_h = 5
                    # Calculate max height needed for this row
                    _max_lines = max(
                        len(pdf.multi_cell(_col_w-2, _row_h, c, split_only=True))
                        if hasattr(pdf,"multi_cell") else 1
                        for c in (_row + [""] * (_ncols - len(_row)))
                    )
                    _rh = _max_lines * _row_h
                    _x0 = pdf.l_margin
                    _y0 = pdf.get_y()
                    if _y0 + _rh > pdf.h - 18:
                        pdf.add_page(); _y0 = pdf.get_y()
                    for _ci, _cell in enumerate((_row + [""] * (_ncols - len(_row)))):
                        pdf.set_xy(_x0 + _ci * _col_w, _y0)
                        pdf.set_draw_color(180,180,180)
                        pdf.set_line_width(0.2)
                        pdf.rect(_x0 + _ci * _col_w, _y0, _col_w, _rh)
                        pdf.set_xy(_x0 + _ci * _col_w + 1, _y0 + 1)
                        pdf.multi_cell(_col_w - 2, _row_h, _cell,
                                       align="C" if _is_hdr else "L",
                                       fill=_is_hdr, border=0)
                    pdf.set_xy(_x0, _y0 + _rh)
            pdf.ln(2); continue

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 16); set_c(C_H1)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(cw, 8, safe(line[2:].strip()), align="L")
            pdf.set_draw_color(*C_H2); pdf.set_line_width(0.5)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3); continue

        # H2
        if line.startswith("## ") and not line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13); set_c(C_H2)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(cw, 7, safe(line[3:].strip()), align="L")
            pdf.ln(1); continue

        # H3
        if line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11); set_c(C_H3)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(cw, 6, safe(line[4:].strip()), align="L")
            pdf.ln(1); continue

        # Numbered list item: "1. Fact text here"
        num_match = _r.match(r'^(\d+)\.\s+(.*)', line)
        if num_match:
            num   = num_match.group(1)
            text  = safe(num_match.group(2))
            pdf.set_font('Helvetica', '', 10); set_c(C_BODY)
            pdf.set_x(pdf.l_margin)
            num_w = pdf.get_string_width(num + '.  ')
            pdf.cell(num_w, 5, num + '.')
            pdf.set_x(pdf.l_margin + num_w)
            pdf.multi_cell(cw - num_w, 5, text, align='L')
            continue

        # Bullet
        if _r.match(r'^[-*]\s+', line):  # bullets already sanitized to '-'
            text = _r.sub(r'^[-*]\s+', '', line)
            pdf.set_font("Helvetica", "", 10); set_c(C_BODY)
            pdf.set_x(pdf.l_margin + 4)
            pdf.cell(5, 5, "-")
            pdf.set_x(pdf.l_margin + 9)
            pdf.multi_cell(cw - 9, 5, safe(text), align="L")
            continue

        # Variable label line  e.g.  some_var: 42 units
        if _r.match(r'^[a-z_][a-z_0-9]*\s*:', line):
            pdf.set_font("Courier", "", 9); set_c(C_CODE)
            pdf.set_fill_color(245, 250, 245)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(cw, 5, sanitize(line), align="L", fill=True)
            continue

        # CONSTRAINT VERIFICATION section divider
        if "CONSTRAINT VERIFICATION" in line.upper():
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 9); set_c(C_GOLD)
            pdf.set_draw_color(*C_GOLD); pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2)
            pdf.multi_cell(cw, 5, "CONSTRAINT VERIFICATION LABELS", align="C")
            pdf.ln(1); continue

        # Regular body
        text = safe(line)
        pdf.set_font("Helvetica", "", 10); set_c(C_BODY)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(cw, 5, text, align="L")

    return bytes(pdf.output())


st.set_page_config(
    page_title="NeuroSymbolic Verifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  (identical design to v2 — only functional changes)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#0c0e14!important;color:#e8e4dc!important;}
.stApp{background:#0c0e14;}
.nsv-header{text-align:center;padding:2.4rem 0 1.4rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.8rem;}
.nsv-logo{font-family:'DM Serif Display',serif;font-size:2.4rem;font-weight:400;letter-spacing:-0.02em;color:#f0ebe0;line-height:1;margin-bottom:0.3rem;}
.nsv-logo span{color:#c8a96e;font-style:italic;}
.nsv-subtitle{font-size:0.75rem;font-weight:300;color:rgba(232,228,220,0.38);letter-spacing:0.12em;text-transform:uppercase;}
.section-label{font-size:0.66rem;letter-spacing:0.14em;text-transform:uppercase;color:#c8a96e;font-weight:600;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;}
.section-label::after{content:'';flex:1;height:1px;background:rgba(200,169,110,0.18);}
.glass-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:1.3rem;margin-bottom:1rem;backdrop-filter:blur(12px);}
.rules-container{display:flex;flex-direction:column;gap:0.4rem;}
.rule-chip{display:flex;align-items:center;gap:0.6rem;background:rgba(200,169,110,0.07);border:1px solid rgba(200,169,110,0.18);border-radius:10px;padding:0.45rem 0.8rem;font-family:'DM Mono',monospace;font-size:0.78rem;color:#e8e4dc;animation:fadeSlide 0.22s ease forwards;}
.rule-chip .rule-num{background:rgba(200,169,110,0.22);color:#c8a96e;border-radius:5px;padding:0.08rem 0.35rem;font-size:0.68rem;font-weight:700;min-width:20px;text-align:center;flex-shrink:0;}
.rule-src{font-size:0.62rem;color:rgba(200,169,110,0.5);font-family:'DM Mono',monospace;background:rgba(200,169,110,0.07);border-radius:4px;padding:0.05rem 0.3rem;margin-left:auto;flex-shrink:0;}
@keyframes fadeSlide{from{opacity:0;transform:translateY(-4px);}to{opacity:1;transform:translateY(0);}}
.stTextArea textarea{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(255,255,255,0.09)!important;border-radius:12px!important;color:#e8e4dc!important;font-family:'DM Sans',sans-serif!important;font-size:0.87rem!important;}
.stTextArea textarea:focus{border-color:rgba(200,169,110,0.45)!important;box-shadow:0 0 0 3px rgba(200,169,110,0.07)!important;}
.stTextInput input{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(255,255,255,0.09)!important;border-radius:10px!important;color:#e8e4dc!important;font-family:'DM Mono',monospace!important;font-size:0.83rem!important;}
.stTextInput input:focus{border-color:rgba(200,169,110,0.45)!important;}
.stButton>button{background:linear-gradient(135deg,#c8a96e 0%,#a8854a 100%)!important;border:none!important;border-radius:11px!important;color:#0c0e14!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:0.87rem!important;padding:0.55rem 1.4rem!important;width:100%;transition:opacity 0.18s,transform 0.13s!important;}
.stButton>button:hover{opacity:0.88!important;transform:translateY(-1px)!important;}
.score-big{font-family:'DM Serif Display',serif;font-size:3.8rem;line-height:1;font-weight:400;margin-bottom:0.2rem;}
.score-pass{color:#6dcea8;}.score-warn{color:#e8c06d;}.score-fail{color:#e8736d;}
.score-label{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:rgba(232,228,220,0.35);}
.audit-row{display:grid;grid-template-columns:6px 1fr;margin-bottom:0.75rem;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);}
.audit-sidebar.pass{background:#6dcea8;}.audit-sidebar.fail{background:#e8736d;}.audit-sidebar.warn{background:#e8c06d;}
.audit-body{padding:0.85rem 1.1rem;background:rgba(255,255,255,0.025);}
.audit-top{display:flex;align-items:flex-start;gap:0.7rem;margin-bottom:0.65rem;}
.audit-badge{font-size:0.66rem;font-weight:700;padding:0.15rem 0.45rem;border-radius:5px;letter-spacing:0.07em;font-family:'DM Mono',monospace;flex-shrink:0;}
.audit-badge.pass{background:rgba(109,206,168,0.15);color:#6dcea8;}
.audit-badge.fail{background:rgba(232,115,109,0.15);color:#e8736d;}
.audit-badge.warn{background:rgba(232,192,109,0.15);color:#e8c06d;}
.audit-rule-text{font-family:'DM Mono',monospace;font-size:0.79rem;color:#e8e4dc;line-height:1.4;flex:1;}
.audit-sym-tag{font-size:0.62rem;color:rgba(200,169,110,0.6);background:rgba(200,169,110,0.08);border-radius:4px;padding:0.1rem 0.3rem;white-space:nowrap;}
.audit-pills{display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.55rem;}
.audit-pill{display:inline-flex;flex-direction:column;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:0.3rem 0.55rem;min-width:80px;}
.ap-label{font-size:0.58rem;text-transform:uppercase;letter-spacing:0.09em;color:rgba(232,228,220,0.3);margin-bottom:0.1rem;}
.ap-value{font-family:'DM Mono',monospace;font-size:0.75rem;color:#d8d4ca;}
.audit-explanation{font-size:0.79rem;color:rgba(232,228,220,0.5);line-height:1.55;padding-top:0.45rem;border-top:1px solid rgba(255,255,255,0.05);}
.audit-domain-warn{font-size:0.74rem;color:#e8c06d;background:rgba(232,192,109,0.07);padding:0.25rem 0.5rem;border-radius:5px;border-left:3px solid #e8c06d;margin-bottom:0.45rem;}
.confidence-bar-wrap{display:flex;align-items:center;gap:0.5rem;margin-top:0.35rem;}
.confidence-bar-bg{flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;}
.confidence-bar-fill{height:100%;border-radius:2px;}
.cb-label{font-size:0.62rem;font-family:'DM Mono',monospace;color:rgba(232,228,220,0.35);white-space:nowrap;}
.iter-badge{display:inline-block;font-size:0.68rem;font-family:'DM Mono',monospace;background:rgba(200,169,110,0.12);color:#c8a96e;border:1px solid rgba(200,169,110,0.25);border-radius:6px;padding:0.1rem 0.45rem;margin-right:0.4rem;}
.iter-score-pass{color:#6dcea8;font-weight:600;}.iter-score-warn{color:#e8c06d;font-weight:600;}.iter-score-fail{color:#e8736d;font-weight:600;}
.diff-add{background:rgba(109,206,168,0.12);color:#6dcea8;display:inline;}
.diff-remove{background:rgba(232,115,109,0.1);color:#e8736d;text-decoration:line-through;display:inline;}
.insight-rule-row{display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0.5rem;border-radius:8px;background:rgba(255,255,255,0.02);margin-bottom:0.25rem;font-size:0.78rem;}
.src-badge{font-size:0.6rem;font-family:'DM Mono',monospace;border-radius:4px;padding:0.08rem 0.35rem;font-weight:600;}
.src-user{background:rgba(200,169,110,0.15);color:#c8a96e;}
.src-doc{background:rgba(109,206,168,0.12);color:#6dcea8;}
.src-wiki{background:rgba(138,110,200,0.14);color:#a88ecc;}
.src-ddg{background:rgba(100,150,232,0.14);color:#6496e8;}
.brain-stat{display:flex;flex-direction:column;align-items:center;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.85rem 1.2rem;text-align:center;}
.brain-stat-num{font-family:'DM Serif Display',serif;font-size:2rem;color:#c8a96e;line-height:1;}
.brain-stat-label{font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(232,228,220,0.35);margin-top:0.2rem;}
.sym-node{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.55rem;}
.sym-node-top{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;}
.sym-type-badge{font-size:0.62rem;font-weight:700;padding:0.12rem 0.4rem;border-radius:5px;font-family:'DM Mono',monospace;}
.sym-type-constraint{background:rgba(200,169,110,0.14);color:#c8a96e;}
.sym-type-observation{background:rgba(109,206,168,0.12);color:#6dcea8;}
.sym-type-source{background:rgba(138,110,200,0.14);color:#a88ecc;}
.sym-type-audit-pass{background:rgba(109,206,168,0.12);color:#6dcea8;}
.sym-type-audit-fail{background:rgba(232,115,109,0.14);color:#e8736d;}
.sym-node-text{font-family:'DM Mono',monospace;font-size:0.77rem;color:#d8d4ca;flex:1;line-height:1.35;}
.sym-meta-row{display:flex;flex-wrap:wrap;gap:0.35rem;}
.sym-meta-chip{font-size:0.62rem;font-family:'DM Mono',monospace;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:5px;padding:0.1rem 0.35rem;color:rgba(232,228,220,0.45);}
.sym-meta-chip.highlight{background:rgba(200,169,110,0.1);border-color:rgba(200,169,110,0.2);color:#c8a96e;}
.sym-ts{font-size:0.6rem;color:rgba(232,228,220,0.22);font-family:'DM Mono',monospace;margin-left:auto;white-space:nowrap;}
.gen-output{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:13px;padding:1.2rem 1.4rem;font-size:0.85rem;line-height:1.78;color:#d0cbbf;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow-y:auto;}
.ref-pill{display:inline-flex;align-items:center;gap:0.3rem;background:rgba(200,169,110,0.07);border:1px solid rgba(200,169,110,0.16);border-radius:20px;padding:0.25rem 0.65rem;font-size:0.72rem;color:#c8a96e;text-decoration:none;}
.pipe-flow{display:flex;align-items:center;justify-content:center;gap:0;padding:0.9rem 0;flex-wrap:wrap;}
.pipe-node{display:flex;flex-direction:column;align-items:center;gap:0.25rem;padding:0 0.35rem;}
.pipe-node .pn-icon{width:38px;height:38px;border-radius:10px;border:1px solid rgba(255,255,255,0.09);display:flex;align-items:center;justify-content:center;font-size:0.95rem;background:rgba(255,255,255,0.03);}
.pipe-node .pn-label{font-size:0.56rem;text-transform:uppercase;letter-spacing:0.09em;color:rgba(232,228,220,0.32);text-align:center;max-width:58px;}
.pipe-arrow{color:rgba(200,169,110,0.32);font-size:0.85rem;padding:0 0.15rem;}
hr{border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.2rem 0;}
[data-testid="metric-container"]{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:13px;padding:0.85rem 1rem;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;font-size:1.8rem!important;color:#e8e4dc;}
[data-testid="stMetricLabel"]{font-size:0.66rem!important;text-transform:uppercase;letter-spacing:0.1em;color:rgba(232,228,220,0.35);}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(255,255,255,0.07);}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:rgba(232,228,220,0.4)!important;border-bottom:2px solid transparent!important;font-size:0.82rem;}
.stTabs [aria-selected="true"]{color:#c8a96e!important;border-bottom-color:#c8a96e!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,#c8a96e,#6dcea8)!important;border-radius:4px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nsv-header">
  <div class="nsv-logo">Neuro<span>Symbolic</span> Verifier</div>
  <div class="nsv-subtitle">v3 · Claude · Graded LTN · Rewrite Loop · Insight Panel</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-panel" style="margin-bottom:1.6rem;">
  <div class="pipe-flow">
    <div class="pipe-node"><div class="pn-icon">📝</div><div class="pn-label">Rules &amp; Prompt</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🌐</div><div class="pn-label">M4 Research</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🧩</div><div class="pn-label">M2 Parser</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🧠</div><div class="pn-label">Qdrant Brain</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">✍️</div><div class="pn-label">Draft Gen</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🔍</div><div class="pn-label">M2 Audit</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">⚖️</div><div class="pn-label">M1 LTN</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🔁</div><div class="pn-label">Rewrite Loop</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">📊</div><div class="pn-label">Verdict</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in [("rules", []), ("results", None), ("input_counter", 0),
               ("qdrant_client", None), ("brain_records", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# If the app crashed mid-run, _pipeline_running stays True and blocks
# the Run button forever. Clear it on every fresh page load if results
# are already stored (meaning the run actually completed before crash).
if st.session_state.get("_pipeline_running", False):
    if st.session_state.results is not None:
        # Run completed — flag just didn't clear
        st.session_state["_pipeline_running"] = False
    # If results is None + flag is True = genuinely mid-run or hard crash
    # The Reset button will clear this manually

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)


def _resolve(env_key: str) -> str:
    try:
        v = st.secrets.get(env_key, "")
        if v: return v.strip()
    except Exception:
        pass
    return os.getenv(env_key, "").strip()


if "resolved_api_key"   not in st.session_state:
    st.session_state.resolved_api_key   = _resolve("ANTHROPIC_API_KEY")
if "resolved_qdrant_url" not in st.session_state:
    st.session_state.resolved_qdrant_url = _resolve("QDRANT_URL")
if "resolved_qdrant_key" not in st.session_state:
    st.session_state.resolved_qdrant_key = _resolve("QDRANT_API_KEY")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT INJECTION HELPERS  (restored from main.py)
# ══════════════════════════════════════════════════════════════════════════════

def _build_constraint_injection(structured_rules: list) -> str:
    """
    Build explicit hard-constraint instructions for the generation prompt.
    Forces the LLM to write `variable: value` labels verbatim so the
    auditor can extract real values — not just audit vague prose.
    """
    lines = ["HARD CONSTRAINTS — YOU MUST SATISFY ALL OF THESE EXACTLY:",
             "(Every constraint is mathematically verified after generation.)\n"]
    for i, rule in enumerate(structured_rules):
        op     = rule.get("operator", "")
        thresh = rule.get("threshold")
        t_low  = rule.get("threshold_low")
        t_high = rule.get("threshold_high")
        unit   = rule.get("unit", "")
        var    = rule.get("variable", "")
        disp   = rule.get("display", rule.get("original", ""))
        ctype  = rule.get("constraint_type", "")

        lines.append(f"  Constraint {i+1}: {disp}")

        if var in ("constraint", "", None):
            lines.append(f"    → REQUIRED: Ensure your response satisfies: {rule.get('original', disp)}")
            lines.append(f"      Explicitly state in your response how this requirement is met.")
        elif op in ("<", "<=") and thresh is not None:
            example = max(0, float(thresh) - 1)
            lines.append(f"    → REQUIRED: Write the exact line `{var}: X {unit}`")
            lines.append(f"      where X <= {thresh}. Example: `{var}: {example:.4g} {unit}`")
        elif op in (">", ">=") and thresh is not None:
            example = float(thresh) + 1
            lines.append(f"    → REQUIRED: Write the exact line `{var}: X {unit}`")
            lines.append(f"      where X > {thresh}. Example: `{var}: {example:.4g} {unit}`")
        elif op == "==" and thresh is not None:
            lines.append(f"    → REQUIRED: Write the exact line `{var}: {thresh} {unit}`")
        elif op == "in_range" and t_low is not None and t_high is not None:
            lines.append(f"    → REQUIRED: Write the exact line `{var}: X {unit}`")
            lines.append(f"      where {t_low} <= X <= {t_high}.")
        elif ctype == "boolean":
            bool_val = "true" if thresh is None or str(thresh).lower() in ("true","1","1.0") else "false"
            lines.append(f"    → REQUIRED: Write this exact line verbatim: `{var}: {bool_val}`")
            lines.append(f"      Prose like 'the system supports X' is NOT sufficient.")
        elif ctype == "categorical" or op in ("contains", "excludes"):
            lines.append(f"    → REQUIRED: Write `{var}: [value]` satisfying: {rule.get('original', disp)}")
        else:
            lines.append(f"    → REQUIRED: Write `{var}: [explicit value]` satisfying: {disp}")
        lines.append("")

    lines.append("IMPORTANT: Use explicit numbers throughout.")
    lines.append("Never use vague terms like 'several', 'a few', 'moderate'.")
    lines.append("Every constraint variable MUST appear with its exact value.")
    lines.append("All boolean label lines (variable_name: true/false) go at the END")
    lines.append("of your response under the header 'CONSTRAINT VERIFICATION LABELS'.")
    return "\n".join(lines)


def _build_violation_feedback(violations: list) -> str:
    if not violations:
        return ""
    lines = ["\n⚠️  YOUR PREVIOUS ATTEMPT VIOLATED THE FOLLOWING CONSTRAINTS:"]
    for v in violations:
        lines.append(f"\n  Rule     : {v['rule_display']}")
        lines.append(f"  Found    : {v['extracted_value_raw']}")
        lines.append(f"  Score    : {v.get('compliance_score', 0):.2f} / 1.0")
        lines.append(f"  Problem  : {v['explanation']}")
        lines.append(f"  FIX THIS : Rewrite so output explicitly satisfies {v['rule_display']}")
    lines.append("\nDo NOT repeat these violations.")
    return "\n".join(lines)


def _dedup_rules(rules: list) -> list:
    """Remove duplicate rules with same (variable, operator, threshold)."""
    seen, deduped = set(), []
    for r in rules:
        key = (r.get("variable",""), r.get("operator",""), str(r.get("threshold","")))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _detect_contradictions(rules: list) -> list:
    """
    Detect logically impossible rule pairs on the same variable.
    Returns list of contradiction dicts: {rule_a, rule_b, reason}.

    Cases checked:
      - upper_bound < lower_bound on same variable (e.g. <=400 AND >=600)
      - lower_bound > upper_bound on same variable
      - in_range where threshold_low > threshold_high
      - == X AND == Y (different values, same variable)
      - == X AND < X  (exact value but also strictly less than it)
      - == X AND > X  (exact value but also strictly greater than it)
    """
    contradictions = []
    # Group rules by variable name
    by_var: dict = {}
    for r in rules:
        v = r.get("variable", "").strip()
        if v:
            by_var.setdefault(v, []).append(r)

    for var, group in by_var.items():
        if len(group) < 2:
            continue
        for i, ra in enumerate(group):
            for rb in group[i+1:]:
                oa, ob = ra.get("operator"), rb.get("operator")
                ta, tb = ra.get("threshold"), rb.get("threshold")
                la, lb = ra.get("threshold_low"), rb.get("threshold_low")
                ha, hb = ra.get("threshold_high"), rb.get("threshold_high")

                try:
                    # Upper bound < lower bound
                    if oa in ("<","<=") and ob in (">",">=") and ta is not None and tb is not None:
                        limit_a = float(ta)
                        limit_b = float(tb)
                        if oa == "<" and ob == ">" and limit_a <= limit_b:
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} < {limit_a} AND {var} > {limit_b}"})
                        elif oa == "<=" and ob == ">=" and limit_a < limit_b:
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} <= {limit_a} AND {var} >= {limit_b}"})
                        elif oa == "<" and ob == ">=" and limit_a <= limit_b:
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} < {limit_a} AND {var} >= {limit_b}"})
                        elif oa == "<=" and ob == ">" and limit_a <= limit_b:
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} <= {limit_a} AND {var} > {limit_b}"})

                    # == X AND == Y (different exact values)
                    if oa == "==" and ob == "==" and ta is not None and tb is not None:
                        if abs(float(ta) - float(tb)) > 1e-9:
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} == {ta} AND {var} == {tb}"})

                    # == X AND strictly outside X
                    if oa == "==" and ta is not None and tb is not None:
                        if ob == "<" and float(ta) >= float(tb):
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} == {ta} AND {var} < {tb}"})
                        if ob == ">" and float(ta) <= float(tb):
                            contradictions.append({"rule_a": ra, "rule_b": rb,
                                "reason": f"Impossible: {var} == {ta} AND {var} > {tb}"})

                    # in_range where bounds are inverted
                    if oa == "in_range" and la is not None and ha is not None:
                        if float(la) > float(ha):
                            contradictions.append({"rule_a": ra, "rule_b": ra,
                                "reason": f"Impossible range: {var} in [{la}, {ha}] — low > high"})

                except (TypeError, ValueError):
                    pass

    # Deduplicate (same pair may be found twice)
    seen_pairs, unique = set(), []
    for c in contradictions:
        key = frozenset([c["rule_a"].get("variable",""), c["rule_b"].get("variable",""),
                          c["reason"]])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique.append(c)
    return unique


def _diff_sentences(text_a: str, text_b: str) -> str:
    """Return a simple word-level diff summary between two drafts."""
    lines_a = text_a.split("\n")
    lines_b = text_b.split("\n")
    diff    = list(difflib.unified_diff(lines_a, lines_b, lineterm="", n=0))
    adds    = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removes = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    return {"added": adds[:6], "removed": removes[:6]}


# ── Source badge helpers — module-level so they're available in ALL tabs ──────
# Defined here (not inside the pipeline block) so they remain in scope when
# Streamlit re-renders the results tabs on subsequent page loads.
_SRC_BADGE_HTML = {
    "User"         : '<span class="src-badge src-user">USER</span>',
    "Document"     : '<span class="src-badge src-doc">DOC</span>',
    "Wikipedia"    : '<span class="src-badge src-wiki">WIKI</span>',
    "DuckDuckGo"   : '<span class="src-badge src-ddg">DDG</span>',
    "Web Search"   : '<span class="src-badge" style="background:rgba(100,180,232,0.14);color:#64b4e8;">WEB</span>',
    "Google Search": '<span class="src-badge" style="background:rgba(80,200,120,0.14);color:#50c878;">GOOG</span>',
    "Custom URL"   : '<span class="src-badge" style="background:rgba(200,150,232,0.14);color:#c896e8;">URL</span>',
    "Research"     : '<span class="src-badge" style="background:rgba(138,110,200,0.14);color:#a88ecc;">RES</span>',
    ""             : '<span class="src-badge src-user">USER</span>',
}

def _src_badge(sn: str) -> str:
    """Return the right HTML badge span for any source name, with smart fallback."""
    if sn in _SRC_BADGE_HTML:
        return _SRC_BADGE_HTML[sn]
    sl = sn.lower()
    if "wiki"   in sl: return _SRC_BADGE_HTML["Wikipedia"]
    if "duck"   in sl or "ddg"    in sl: return _SRC_BADGE_HTML["DuckDuckGo"]
    if "google" in sl:                   return _SRC_BADGE_HTML["Google Search"]
    if "web"    in sl or "search" in sl: return _SRC_BADGE_HTML["Web Search"]
    if "url"    in sl or "http"   in sl: return _SRC_BADGE_HTML["Custom URL"]
    if "doc"    in sl or "upload" in sl: return _SRC_BADGE_HTML["Document"]
    abbr = sn.upper()[:5] if sn else "SRC"
    return (f'<span class="src-badge" style="background:rgba(180,180,180,0.12);'
            f'color:rgba(232,228,220,0.55);">{abbr}</span>')

_SRC_COLORS = {
    "User":"src-user","Document":"src-doc",
    "Wikipedia":"src-wiki","DuckDuckGo":"src-ddg",
    "Web Search":"src-ddg","Google Search":"src-wiki",
    "Custom URL":"src-doc","Research":"src-wiki","":"src-user",
}

def _insight_sc(sn: str) -> str:
    """Return a CSS class for the insight tab badge."""
    if sn in _SRC_COLORS: return _SRC_COLORS[sn]
    sl = sn.lower()
    if "wiki"   in sl: return "src-wiki"
    if "duck"   in sl or "ddg"    in sl: return "src-ddg"
    if "doc"    in sl or "url"    in sl: return "src-doc"
    return "src-user"


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:

    # ── LLM Provider + Model + API Key ───────────────────────────────────────
    st.markdown('<div class="section-label">🤖 LLM Provider</div>', unsafe_allow_html=True)

    # ── Thinking models: need special API handling ───────────────────────────
    # Claude: extended thinking via betas header + thinking block in messages
    # OpenAI o-series: reasoning models — use same API, no system prompt, no temp
    # Gemini 3.x: thinking built-in, no extra params needed
    _THINKING_MODELS = {
        # Claude extended thinking
        "claude-opus-4-6", "claude-sonnet-4-6",
        # OpenAI reasoning (o-series)
        "o3", "o3-pro", "o4-mini",
    }

    _PROVIDERS = {
        "Anthropic (Claude)": {
            "id": "anthropic",
            "models": [
                "claude-opus-4-6",          # Latest flagship (Feb 2026) — extended thinking
                "claude-sonnet-4-6",         # Latest balanced (Feb 2026) — extended thinking
                "claude-sonnet-4-5",         # Previous balanced
                "claude-opus-4-5",           # Previous flagship
                "claude-haiku-4-5-20251001", # Fast / low cost
            ],
            "key_hint" : "sk-ant-…",
            "env_key"  : "ANTHROPIC_API_KEY",
            "resolved" : st.session_state.resolved_api_key,
        },
        "OpenAI (GPT)": {
            "id": "openai",
            "models": [
                "gpt-5.4",       # Latest flagship (2026)
                "gpt-5.4-mini",  # Fast / efficient
                "gpt-4.1",       # Strong coding + 1M context
                "gpt-4.1-mini",  # Cost-efficient
                "o3",            # Reasoning model (thinking)
                "o3-pro",        # Reasoning — max compute (thinking)
                "o4-mini",       # Reasoning — efficient (thinking)
            ],
            "key_hint" : "sk-…",
            "env_key"  : "OPENAI_API_KEY",
            "resolved" : _resolve("OPENAI_API_KEY"),
        },
        "Google (Gemini)": {
            "id": "google",
            "models": [
                "gemini-3-flash",            # Gemini 3 Flash (Mar 2026) — fastest 3-series
                "gemini-3.1-pro-preview",    # Gemini 3.1 Pro (Feb 2026) — flagship
                "gemini-3.1-flash-preview",  # Gemini 3.1 Flash — balanced
                "gemini-3.1-flash-lite",     # Gemini 3.1 Flash-Lite — cost-efficient
                "gemini-2.5-pro",            # Stable premium
                "gemini-2.5-flash",          # Stable balanced
            ],
            "key_hint" : "AIza…",
            "env_key"  : "GOOGLE_API_KEY",
            "resolved" : _resolve("GOOGLE_API_KEY"),
        },
    }

    _provider_name = st.selectbox(
        "provider_select", label_visibility="collapsed",
        options=list(_PROVIDERS.keys()), key="provider_select",
    )
    _prov = _PROVIDERS[_provider_name]

    _model_choice = st.selectbox(
        "model_select", label_visibility="collapsed",
        options=_prov["models"], key=f"model_select_{_prov['id']}",
    )
    if _model_choice in _THINKING_MODELS:
        st.markdown(
            '<p style="font-size:0.7rem;color:rgba(200,169,110,0.8);margin-top:-0.3rem;">'
            '🧠 Thinking / reasoning model selected — extended reasoning enabled</p>',
            unsafe_allow_html=True
        )

    _hint = "Auto-loaded ✓" if _prov["resolved"] else _prov["key_hint"]
    api_input = st.text_input(
        f"{_provider_name} API Key", label_visibility="collapsed",
        type="password", placeholder=_hint, key="api_key_field"
    )
    api_key = api_input.strip() or _prov["resolved"]
    if _prov["resolved"] and not api_input.strip():
        st.markdown('<p style="font-size:0.7rem;color:rgba(109,206,168,0.7);margin-top:-0.25rem;">🔒 Loaded from environment</p>', unsafe_allow_html=True)
    elif not api_key:
        st.markdown('<p style="font-size:0.7rem;color:rgba(232,115,109,0.7);margin-top:-0.25rem;">⚠ No key found — paste above</p>', unsafe_allow_html=True)

    # Build the llm_config dict passed through the entire pipeline
    llm_config = {
        "provider"  : _prov["id"],
        "model"     : _model_choice,
        "api_key"   : api_key or "",
        "thinking"  : _model_choice in _THINKING_MODELS,
    }

    # ── Qdrant credentials — loaded silently from m3_vector_db module constants
    # The UI expander has been removed since credentials are baked into m3_vector_db.py.
    # To change the Qdrant instance, edit QDRANT_URL / QDRANT_API_KEY in that file.
    try:
        import m3_vector_db as _m3_cfg
        _m3_qdrant_url = _m3_cfg.QDRANT_URL
        _m3_qdrant_key = _m3_cfg.QDRANT_API_KEY
    except Exception:
        _m3_qdrant_url = ""
        _m3_qdrant_key = ""
    qdrant_url = (st.session_state.resolved_qdrant_url or _m3_qdrant_url or None)
    qdrant_key = (st.session_state.resolved_qdrant_key or _m3_qdrant_key or None)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚙️ Pipeline Settings</div>', unsafe_allow_html=True)
    col_m, col_a = st.columns(2)
    with col_m:
        mode = st.radio("Mode", label_visibility="collapsed",
                        options=["🔬 Full", "📐 Audit Only", "🌐 Research+Gen"],
                        key="pipeline_mode")
    with col_a:
        max_attempts = st.number_input("Max rewrites", min_value=1, max_value=10,
                                        value=5, key="max_attempts_input",
                                        help="Max rewrite iterations before halting")

    ltn_threshold = st.slider("LTN pass threshold", min_value=0.30, max_value=0.99,
                               value=0.80, step=0.05, key="ltn_threshold_slider",
                               help="Score above this = all constraints satisfied")

    # ── Modular Research Sources ─────────────────────────────────────────────
    st.markdown('<div class="section-label">🔍 Research Sources</div>', unsafe_allow_html=True)
    _has_doc = bool(st.session_state.get("doc_upload") or
                    st.session_state.get("existing_draft","").strip())

    # Row 1: Wikipedia + DuckDuckGo (auto-off when doc present)
    _src_r1c1, _src_r1c2 = st.columns(2)
    with _src_r1c1:
        use_wikipedia = st.checkbox("📖 Wikipedia", value=not _has_doc,
                                    key="use_wikipedia",
                                    help="Search Wikipedia and derive rules from the best matching article.")
    with _src_r1c2:
        use_duckduckgo = st.checkbox("🦆 DuckDuckGo (instant)", value=not _has_doc,
                                     key="use_duckduckgo",
                                     help="DuckDuckGo instant answer API — fast topic snippet.")

    # Row 2: Full web search options
    _src_r2c1, _src_r2c2 = st.columns(2)
    with _src_r2c1:
        use_web_search = st.checkbox(
            "🌍 Web Search (DuckDuckGo, top 5)",
            value=False, key="use_web_search_full",
            help="Full web search via DuckDuckGo — fetches and reads top 5 pages. Prioritises authoritative sources (.gov .edu arXiv Reuters Nature). Requires: pip install duckduckgo-search"
        )
    with _src_r2c2:
        use_google_search = st.checkbox(
            "🔍 Google Search (top 5)",
            value=False, key="use_google_search",
            help="Google Custom Search JSON API — fetches and reads top 5 results. Requires a Google API Key + Custom Search Engine ID."
        )

    # Google credentials (shown when Google selected)
    if use_google_search:
        _gc1, _gc2 = st.columns(2)
        with _gc1:
            st.text_input("Google API Key", type="password", placeholder="AIza...",
                          key="google_api_key",
                          help="Create at console.developers.google.com → Custom Search JSON API")
        with _gc2:
            st.text_input("Search Engine ID (cx)", placeholder="abc123:xyz...",
                          key="google_cx",
                          help="Create at programmablesearchengine.google.com — set to search the whole web")
        st.caption("🔍 Google Search fetches top 5 results from across the web.")

    # Row 3: Custom URLs
    use_custom_urls = st.checkbox("🔗 Custom URLs / Online PDFs", value=False,
                                  key="use_custom_urls",
                                  help="Fetch specific URLs and extract text. Supports HTML pages and online PDFs.")
    _custom_url_list = []
    if use_custom_urls:
        _url_raw = st.text_area(
            "custom_urls", label_visibility="collapsed",
            placeholder="One URL per line:\nhttps://arxiv.org/pdf/2310.12345.pdf\nhttps://en.wikipedia.org/wiki/...",
            height=70, key="custom_urls_input"
        )
        _custom_url_list = [u.strip() for u in _url_raw.splitlines() if u.strip().startswith("http")]
        if _custom_url_list:
            # Warn about video/JS-only URLs before the run starts
            _VIDEO_WARN_DOMAINS = {
                "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                "twitch.tv", "tiktok.com", "twitter.com", "x.com",
                "instagram.com", "facebook.com", "linkedin.com",
            }
            import urllib.parse as _up
            _bad_urls = [
                u for u in _custom_url_list
                if any(
                    _up.urlparse(u).netloc.lower().lstrip("www.") == d
                    or _up.urlparse(u).netloc.lower().lstrip("www.").endswith("." + d)
                    for d in _VIDEO_WARN_DOMAINS
                )
            ]
            if _bad_urls:
                st.warning(
                    f"⚠️ **{len(_bad_urls)} unsupported URL(s) detected:** "
                    f"Video platforms (YouTube, Vimeo, TikTok) and social media "
                    f"sites cannot be scraped for text — the system cannot watch "
                    f"or transcribe videos. These URLs will be skipped.\n\n"
                    f"**To use video content:** paste the transcript or description "
                    f"directly into the Reference Document field above.",
                    icon="🎥",
                )
            _ok_urls = [u for u in _custom_url_list if u not in _bad_urls]
            if _ok_urls:
                st.caption(f"✅ {len(_ok_urls)} URL(s) queued — fetched concurrently at run time")

    use_web_research = (use_wikipedia or use_duckduckgo or use_web_search
                        or use_google_search or bool(_custom_url_list))

    if _has_doc and use_web_research:
        st.caption("⚠️ Document loaded + web research on. Research uses your prompt — "
                   "may pull unrelated rules. Disable all sources for pure document tasks.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generation Prompt ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Generation Prompt</div>', unsafe_allow_html=True)
    user_prompt = st.text_area("prompt", label_visibility="collapsed",
                                placeholder="e.g. Write a weekly study plan to improve SAT Math from 600 to 750 in 8 weeks.",
                                height=100, key="user_prompt")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reference Document / Existing Draft ─────────────────────────────────
    st.markdown('<div class="section-label">📄 Reference Document (optional)</div>', unsafe_allow_html=True)

    _doc_tab_paste, _doc_tab_upload = st.tabs(["✏️ Paste text", "📎 Upload file"])

    with _doc_tab_paste:
        _pasted = st.text_area("draft", label_visibility="collapsed",
                               placeholder="Paste an existing draft or reference document here.\n"
                                           "Used as context for generation, or audited directly.",
                               height=90, key="existing_draft")

    with _doc_tab_upload:
        _uploaded_file = st.file_uploader(
            "Upload a document",
            label_visibility="collapsed",
            type=["txt", "pdf", "docx", "md", "csv"],
            key="doc_upload",
            help="Supported: TXT, PDF, DOCX, MD, CSV. Max 50 MB."
        )
        _extracted_text = ""
        if _uploaded_file is not None:
            try:
                _fname = _uploaded_file.name.lower()
                if _fname.endswith(".txt") or _fname.endswith(".md"):
                    _extracted_text = _uploaded_file.read().decode("utf-8", errors="replace")

                elif _fname.endswith(".csv"):
                    import csv, io
                    _csv_reader = csv.reader(io.StringIO(_uploaded_file.read().decode("utf-8", errors="replace")))
                    _extracted_text = "\n".join(", ".join(row) for row in _csv_reader)

                elif _fname.endswith(".pdf"):
                    try:
                        import pypdf
                        _pdf_reader = pypdf.PdfReader(io.BytesIO(_uploaded_file.read()))
                        _extracted_text = "\n\n".join(
                            page.extract_text() or "" for page in _pdf_reader.pages
                        ).strip()
                    except ImportError:
                        try:
                            import pdfplumber
                            with pdfplumber.open(io.BytesIO(_uploaded_file.read())) as _plumb:
                                _extracted_text = "\n\n".join(
                                    p.extract_text() or "" for p in _plumb.pages
                                ).strip()
                        except ImportError:
                            st.warning("PDF support requires pypdf. Run: pip install pypdf")

                elif _fname.endswith(".docx"):
                    try:
                        import docx as _docx_lib
                        import io
                        _doc = _docx_lib.Document(io.BytesIO(_uploaded_file.read()))
                        _extracted_text = "\n".join(
                            para.text for para in _doc.paragraphs if para.text.strip()
                        )
                    except ImportError:
                        st.warning("DOCX support requires python-docx. Run: pip install python-docx")

                if _extracted_text:
                    _preview = _extracted_text[:300].replace("<","&lt;")
                    st.markdown(
                        f'<div style="background:rgba(109,206,168,0.06);border:1px solid rgba(109,206,168,0.2);"'
                        f'border-radius:10px;padding:0.7rem 1rem;font-size:0.78rem;color:rgba(232,228,220,0.65);"'
                        f'font-family:DM Mono,monospace;max-height:80px;overflow:hidden;">'
                        f'✅ <strong>{_uploaded_file.name}</strong> — {len(_extracted_text):,} chars extracted<br>'
                        f'<span style="opacity:0.5">{_preview}{"…" if len(_extracted_text)>300 else ""}</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Could not extract text from this file.")
            except Exception as _upload_err:
                st.error(f"File read error: {_upload_err}")

    # Merge: uploaded file takes priority over pasted text if both present
    existing_draft = _extracted_text.strip() if _extracted_text.strip() else _pasted.strip()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Constraint Rules ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📏 Constraint Rules</div>', unsafe_allow_html=True)
    ta_key = f"rule_input_{st.session_state.input_counter}"
    st.text_area("rules_raw", label_visibility="collapsed",
                 placeholder="One rule per line:\n  • Study sessions ≤ 2 hours each\n  • Weekly tests ≥ 2\n  • Total weekly hours ≤ 14",
                 height=110, key=ta_key)

    btn_add, btn_clear = st.columns([1, 1])
    with btn_add:
        if st.button("＋ Add Rule(s)", key="add_rule_btn"):
            raw = st.session_state.get(ta_key, "").strip()
            if raw:
                for line in raw.split("\n"):
                    line = line.strip().lstrip("-•*›▸").strip()
                    if line and line not in st.session_state.rules:
                        st.session_state.rules.append(line)
            st.session_state.input_counter += 1
            st.rerun()
    with btn_clear:
        if st.button("✕ Clear All", key="clear_rules_btn"):
            st.session_state.rules = []
            st.rerun()

    if st.session_state.rules:
        chips = '<div class="rules-container" style="margin-top:0.7rem;">'
        for i, r in enumerate(st.session_state.rules):
            e = r.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            chips += f'<div class="rule-chip"><span class="rule-num">R{i+1}</span><span style="flex:1">{e}</span><span class="rule-src">User</span></div>'
        chips += '</div>'
        st.markdown(chips, unsafe_allow_html=True)
        with st.expander("🗑 Remove individual rules"):
            for i, r in enumerate(list(st.session_state.rules)):
                lbl = r[:55] + ("…" if len(r) > 55 else "")
                if st.button(f"Remove R{i+1}: {lbl}", key=f"del_{i}"):
                    st.session_state.rules.pop(i); st.rerun()
    else:
        st.markdown('<div style="text-align:center;color:rgba(232,228,220,0.2);font-size:0.78rem;padding:0.9rem;border:1px dashed rgba(255,255,255,0.06);border-radius:10px;margin-top:0.4rem;">No rules yet — type above and click Add</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _run_col, _rst_col = st.columns([3, 1])
    with _run_col:
        run_btn = st.button("⚡ Run Pipeline", key="run_pipeline_btn",
                            use_container_width=True)
    with _rst_col:
        _hard_reset = st.button("🔄 Reset", key="hard_reset_btn",
                                use_container_width=True,
                                help="Clear all results and pipeline state. Use if the run gets stuck or after an API error.")
    if _hard_reset:
        # Wipe everything — results, running flag, any stuck state
        _keys_to_clear = [
            "results", "brain_records", "_pipeline_running",
        ]
        for _k in _keys_to_clear:
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state.results       = None
        st.session_state.brain_records = {}
        st.session_state["_pipeline_running"] = False
        # Reset the Qdrant client singleton so the next run starts with
        # a completely fresh connection and empty in-memory store.
        # This is the key fix for cross-query contamination — without this,
        # the module-level _CLIENT retains all data from prior runs.
        try:
            import m3_vector_db as _m3_reset
            _m3_reset.reset_client()
        except Exception:
            pass
        st.success("✅ Reset complete — ready for a new run.", icon="🔄")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_right:

    if not run_btn and st.session_state.results is None:
        st.markdown("""
        <div class="glass-panel" style="text-align:center;padding:2.8rem 2rem;">
            <div style="font-size:2.6rem;margin-bottom:0.8rem;opacity:0.3;">⚖️</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:rgba(232,228,220,0.4);margin-bottom:0.4rem;">Awaiting verification</div>
            <div style="font-size:0.78rem;color:rgba(232,228,220,0.2);line-height:1.65;">Configure prompt, rules &amp; API key<br>then press <strong style="color:rgba(200,169,110,0.4)">Run Pipeline</strong></div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE EXECUTION
    # ══════════════════════════════════════════════════════════════════════════
    if run_btn and not st.session_state.get('_pipeline_running', False):
        st.session_state['_pipeline_running'] = True
        if not api_key.strip():
            st.error("⚠️ Please enter your Anthropic API key."); st.stop()
        if not user_prompt.strip() and not existing_draft.strip():
            st.error("⚠️ Please enter a prompt or paste an existing draft."); st.stop()

        try:
            import anthropic as _anthropic
        except ImportError:
            st.error("`anthropic` not installed. Run: pip install anthropic"); st.stop()

        try:
            import m2_llm_parser as m2
        except ImportError as e:
            st.error(f"Cannot import m2_llm_parser: {e}"); st.stop()

        try:
            import m3_vector_db as m3
            has_m3 = True
        except ImportError:
            has_m3 = False
            st.warning("⚠️ qdrant-client or sentence-transformers unavailable — memory step skipped.")

        try:
            import m4_agentic_router as m4
            has_m4 = True
        except ImportError:
            has_m4 = False

        try:
            import m1_ltn_core as m1
            has_ltn = True
        except ImportError:
            has_ltn = False

        prog   = st.progress(0, text="Initialising…")
        status = st.empty()
        results = {}
        run_id  = str(uuid.uuid4())[:8]
        rules_snapshot = list(st.session_state.rules)

        # ── Init Qdrant ───────────────────────────────────────────────────────
        qdrant_client = None
        if has_m3:
            try:
                qdrant_client = m3.setup_memory(url=qdrant_url, qdrant_api_key=qdrant_key)
                st.session_state.qdrant_client = qdrant_client
            except Exception as e:
                st.warning(f"Qdrant init failed (non-fatal): {e}")
                has_m3 = False

        # ── PARALLEL: Research + Rule Parsing ─────────────────────────────────
        source_results   = []
        structured_rules = []
        memory_context   = []
        _web_research_enabled = st.session_state.get("use_web_research", True)
        _doc_present = bool(existing_draft.strip())

        # Safe defaults — prevent NameError if session state keys are missing
        # (can happen on first load or version mismatch during deployment)
        if "_research_config" not in dir():
            _research_config = {
                "wikipedia"      : st.session_state.get("use_wikipedia", not _doc_present),
                "duckduckgo"     : st.session_state.get("use_duckduckgo", not _doc_present),
                "google"         : st.session_state.get("use_google_search", False),
                "google_api_key" : st.session_state.get("google_api_key", "").strip(),
                "google_cx"      : st.session_state.get("google_cx", "").strip(),
                "web_search"     : st.session_state.get("use_web_search_full", False),
                "custom_urls"    : [u.strip() for u in
                                    st.session_state.get("custom_urls_input","").splitlines()
                                    if u.strip().startswith("http")],
            }
        if "_any_research" not in dir():
            _any_research = any([
                _research_config.get("wikipedia", True),
                _research_config.get("duckduckgo", True),
                _research_config.get("google", False),
                _research_config.get("web_search", False),
                bool(_research_config.get("custom_urls", [])),
            ])
        _research_config = {
            "wikipedia"  : st.session_state.get("use_wikipedia", not _doc_present),
            "duckduckgo" : st.session_state.get("use_duckduckgo", not _doc_present),
            "google"         : st.session_state.get("use_google_search", False),
            "google_api_key" : st.session_state.get("google_api_key", "").strip(),
            "google_cx"      : st.session_state.get("google_cx", "").strip(),
            "web_search"     : st.session_state.get("use_web_search_full", False),
            "custom_urls": [u.strip() for u in
                            st.session_state.get("custom_urls_input","").splitlines()
                            if u.strip().startswith("http")],
        }
        _any_research = any([
            _research_config["wikipedia"],
            _research_config["duckduckgo"],
            _research_config["google"],
            _research_config["web_search"],
            bool(_research_config["custom_urls"]),
        ])
        do_research = (
            has_m4
            and mode in ["🔬 Full", "🌐 Research+Gen"]
            and _any_research
        )
        do_rules    = bool(rules_snapshot)

        # Warn if user has research sources checked but mode blocks research
        if _any_research and not do_research and mode == "📐 Audit Only":
            st.warning(
                "⚠️ **Research sources are checked but mode is 'Audit Only'** — "
                "web research is skipped in Audit Only mode. "
                "Switch to **🔬 Full** or **🌐 Research+Gen** to use web research.",
                icon="🔍"
            )

        prog.progress(8, text="⚡ Research & rule parsing in parallel…")

        def _run_research():
            if not user_prompt.strip():
                return []
            return m4.research_all_sources(
                user_prompt.strip(), api_key=api_key,
                llm_config=llm_config, research_config=_research_config,
            )

        def _run_rule_parse():
            return m2.parse_rules_parallel(rules_snapshot, api_key, llm_config=llm_config)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            if do_research: futures["research"] = pool.submit(_run_research)
            if do_rules:    futures["rules"]    = pool.submit(_run_rule_parse)

            msgs = []
            if "research" in futures:
                _rc = _research_config if "_research_config" in dir() else {}
                _srcs = [k for k,v in [
                    ("Wikipedia",  _rc.get("wikipedia")),
                    ("DuckDuckGo", _rc.get("duckduckgo")),
                    ("Google",     _rc.get("google")),
                    ("Web Search", _rc.get("web_search")),
                ] if v] + ([f"{len(_rc.get('custom_urls',[]))} URL(s)"] if _rc.get("custom_urls") else [])
                msgs.append("M4 researching: " + (", ".join(_srcs) if _srcs else "sources"))
            if "rules"    in futures: msgs.append(f"M2 parsing {len(rules_snapshot)} rule(s)")
            if msgs: status.info(" · ".join(msgs) + "…")

            if "research" in futures:
                try:
                    _raw_sources   = futures["research"].result()
                    # Filter out failed/empty fetch results — only keep genuine sources
                    source_results = [s for s in _raw_sources
                                      if m4._is_valid_source(s)]
                    _skipped = len(_raw_sources) - len(source_results)
                    if source_results:
                        print(f"   [App] {len(source_results)} valid source(s) from {len(_raw_sources)} fetched ({_skipped} failed/empty filtered).")
                    elif _raw_sources:
                        # Research ran but ALL results were failures — tell the user
                        st.warning(
                            f"⚠️ **Web research ran but returned no usable content.** "
                            f"{len(_raw_sources)} fetch attempt(s) all failed or returned empty pages. "
                            f"The pipeline will continue using only your rules. "
                            f"Try enabling different research sources or check your internet connection.",
                            icon="🌐"
                        )
                    results["sources"] = source_results
                except Exception as e:
                    st.warning(f"M4 research failed (non-fatal): {e}")
            elif not _any_research if "_any_research" in dir() else True:
                _skip_reason = "document mode" if _doc_present else "all sources disabled"
                status.info(f"📄 Skipping web research ({_skip_reason}) — rules from user + document only.")

            if "rules" in futures:
                try:
                    parsed_user_rules = futures["rules"].result()
                except Exception as e:
                    st.error(f"Rule parsing failed: {e}"); st.stop()
            else:
                parsed_user_rules = []

        status.empty()

        # ── Research → Rule derivation ─────────────────────────────────────────
        research_rules = []
        if source_results and mode == "🔬 Full":
            prog.progress(22, text="🔬 Deriving rules from research…")
            status.info("Deriving constraints from research sources…")
            for src in source_results:
                src_ctx   = src.get("context", "")
                src_name  = src.get("source_name", "Research")
                src_ref   = src.get("reference", "")
                src_title = src.get("title", "the topic")
                derivation_prompt = (
                    f"Based on this {src_name} excerpt about '{src_title}':\n\n"
                    f"'{src_ctx}'\n\n"
                    f"The user is working on: '{user_prompt}'\n\n"
                    f"Derive ONLY constraints a domain expert would require for THIS task.\n"
                    f"Each rule must: be directly relevant, be numerical or boolean, have a "
                    f"CONCRETE threshold (no undefined variables like 'fast_threshold').\n"
                    f"Return a JSON array of rule strings ONLY. Return [] if no rules pass.\n"
                    f"Example good rule: 'yield strength safety factor must be greater than 3.0'\n"
                    f"Example bad rule: 'design must be good' (unverifiable)"
                )
                try:
                    raw_rt = m2._call_llm(derivation_prompt, llm_config, max_tokens=512).strip().replace("```json","").replace("```","").strip()
                    rule_texts = json.loads(raw_rt)
                    if isinstance(rule_texts, list):
                        for rt in rule_texts:
                            parsed = m2.parse_rule_to_constraint(rt, api_key)
                            parsed["source_name"] = src_name
                            parsed["source"]      = src_ref
                            research_rules.append(parsed)
                            print(f"   [Research rule] {rt[:70]}")
                except Exception as e:
                    print(f"   [Research derivation failed] {e}")
            status.empty()

        # Merge user rules + research rules, attach source labels, deduplicate
        for r in parsed_user_rules:
            if r and "source_name" not in r:
                r["source_name"] = "User"
        all_rules = [r for r in parsed_user_rules if r] + research_rules
        structured_rules = _dedup_rules(all_rules)
        results["structured_rules"] = structured_rules

        # ── Contradiction check + auto-resolution ────────────────────────
        _contradictions = _detect_contradictions(structured_rules)
        results["contradictions"] = _contradictions

        if _contradictions:
            status.warning(
                f"⚠️ {len(_contradictions)} contradiction(s) detected — auto-resolving…"
            )
            _resolved_log = []
            _rules_to_remove = set()

            for _ctr in _contradictions:
                _ra = _ctr["rule_a"]
                _rb = _ctr["rule_b"]
                _reason = _ctr["reason"]

                # Strategy: keep the rule from the user (source_name == "User")
                # and remove the research-derived conflicting rule.
                # If both are user rules, keep the more permissive one
                # (upper bound preferred over lower when they conflict)
                # and log the resolution.
                _ra_user = _ra.get("source_name","") == "User"
                _rb_user = _rb.get("source_name","") == "User"

                if _ra_user and not _rb_user:
                    # Remove the research rule
                    _remove_idx = id(_rb)
                    _keep = _ra
                    _drop = _rb
                elif _rb_user and not _ra_user:
                    _remove_idx = id(_ra)
                    _keep = _rb
                    _drop = _ra
                else:
                    # Both user rules or both research — remove the stricter one
                    # (the one with the smaller allowed range / higher lower-bound)
                    _oa = _ra.get("operator","")
                    _ob = _rb.get("operator","")
                    if _oa in ("<","<="):
                        # ra is upper bound — keep it (it allows more)
                        _remove_idx = id(_rb)
                        _keep = _ra; _drop = _rb
                    else:
                        _remove_idx = id(_ra)
                        _keep = _rb; _drop = _ra

                _rules_to_remove.add(id(_drop))
                _resolved_log.append(
                    f"Removed '{_drop.get('display',_drop.get('original','?'))}' "
                    f"(conflicts with '{_keep.get('display',_keep.get('original','?'))}') "
                    f"— {_reason}"
                )

            # Apply removals
            _before = len(structured_rules)
            structured_rules = [r for r in structured_rules
                                 if id(r) not in _rules_to_remove]
            results["structured_rules"] = structured_rules
            results["contradictions_resolved"] = _resolved_log

            _removed_n = _before - len(structured_rules)
            status.info(
                f"✅ Auto-resolved {_removed_n} contradicting rule(s). "
                f"Removed: " + "; ".join(_resolved_log)
            )

        # ── M3: Store in Qdrant ───────────────────────────────────────────────
        brain_records = {"rules": [], "sources": [], "audit": []}
        if has_m3 and qdrant_client:
            prog.progress(32, text="🧠 Storing in Qdrant…")
            status.info("Storing symbolic references…")
            try:
                if source_results:
                    for src in source_results:
                        m3.store_source(qdrant_client, src, run_id=run_id)
                    brain_records["sources"] = m3.get_all_records(
                        qdrant_client, "sources", run_id=run_id)
                if structured_rules:
                    m3.store_all_rules(qdrant_client, structured_rules, run_id=run_id)
                    brain_records["rules"] = m3.get_all_records(
                        qdrant_client, "rules", run_id=run_id)
                query          = user_prompt or existing_draft
                # IMPORTANT: pass run_id so retrieve_context only pulls
                # context from THIS run — prevents cross-query contamination
                memory_context = m3.retrieve_context(
                    qdrant_client, query, n_results=4, run_id=run_id)
                results["memory_context"] = memory_context
            except Exception as e:
                st.warning(
                    f"⚠️ Qdrant step failed (non-fatal): {e}\n\n"
                    f"Second Brain will be empty for this run. "
                    f"The pipeline will continue normally."
                )
            status.empty()

        # Show the user what rules are going to be enforced (with source badges)
        if structured_rules:
            prog.progress(38, text="📋 Rules ready…")
            chips = '<div class="rules-container" style="margin-bottom:0.8rem;">'
            for i, r in enumerate(structured_rules):
                sn     = r.get("source_name", "User")
                bg     = _src_badge(sn)
                disp   = r.get("display", r.get("original",""))[:70]
                disp_e = disp.replace("&","&amp;").replace("<","&lt;")
                chips += f'<div class="rule-chip"><span class="rule-num">R{i+1}</span><span style="flex:1">{disp_e}</span>{bg}</div>'
            chips += '</div>'
            status.markdown(chips, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # REWRITE LOOP  (the core system claim — restored)
        # ══════════════════════════════════════════════════════════════════════
        MAX_ITER         = int(max_attempts)
        LTN_THRESHOLD    = float(ltn_threshold)
        constraint_block = _build_constraint_injection(structured_rules) if structured_rules else ""

        passed               = False
        attempt              = 0
        draft_text           = existing_draft.strip()
        violation_feedback   = ""
        iteration_history    = []   # list of {attempt, draft, audit, ltn_score, violations}
        final_audit_results  = []
        final_ltn_score      = 0.0
        final_draft          = ""

        # If user pasted existing draft → audit only, no generation
        audit_only = bool(existing_draft.strip()) and not user_prompt.strip()

        while (attempt < MAX_ITER) and not passed:
            attempt += 1
            iter_label = f"Iteration {attempt}/{MAX_ITER}"
            prog.progress(min(99, 45 + attempt * 8), text=f"✍️ {iter_label} -- generating...")

            # ── Generate draft (skip if audit-only) ───────────────────────────
            if not audit_only:
                # Build numbered source reference list so the LLM can cite by
                # [Source N] in its draft.  Include title + URL for each source.
                ctx_parts = []
                _src_ref_lines = []   # Used in citation instruction below
                for _si, s in enumerate(source_results):
                    _sn    = s.get("source_name", "Source")
                    _title = s.get("title", "")
                    _url   = (s.get("reference","") or s.get("source","") or "").strip()
                    _ctx   = s.get("context","")
                    _label = f"[Source {_si+1}]"
                    _hdr   = f"{_label} {_sn}"
                    if _title: _hdr += f" — {_title}"
                    if _url and _url not in ("None","none",""):
                        _hdr += f"\nURL: {_url}"
                    ctx_parts.append(f"{_hdr}\n{_ctx}")
                    # Build citation reference line for the instruction
                    _ref_str = f"  {_label} {_sn}"
                    if _title: _ref_str += f" — {_title}"
                    if _url and _url not in ("None","none",""):
                        _ref_str += f" ({_url})"
                    _src_ref_lines.append(_ref_str)

                if memory_context:
                    ctx_parts.append("Relevant constraints:\n" + "\n".join(memory_context))

                # Build document block — injected into prompt when user uploaded/pasted a doc
                _doc_text = existing_draft.strip()
                if _doc_text:
                    _doc_char_limit = 40000
                    _truncated      = len(_doc_text) > _doc_char_limit
                    _doc_preview    = _doc_text[:_doc_char_limit]

                    # Warn user if document was truncated
                    if _truncated:
                        st.warning(
                            f"⚠️ **Document truncated:** Your document is "
                            f"{len(_doc_text):,} characters but the LLM context "
                            f"limit is {_doc_char_limit:,} chars. Only the first "
                            f"{_doc_char_limit:,} characters were sent. "
                            f"Consider splitting the document or reducing other content."
                        )

                    # Warn if document looks like a failed PDF extraction (blank/near-blank)
                    _non_blank_chars = len(_doc_text.replace(" ","").replace("\n",""))
                    if _non_blank_chars < 100:
                        st.warning(
                            "⚠️ **Document appears empty or unreadable.** "
                            "If you uploaded a scanned PDF, the pages are images — "
                            "text extraction returns nothing. Try copy-pasting the "
                            "text directly into the Reference Document field instead."
                        )

                    # Detect whether this is a Q&A scenario (prompt asks something
                    # about the document) vs a generation scenario (document is
                    # background material for a new piece of content).
                    _qa_keywords = (
                        "summarize", "summary", "explain", "what does", "what is",
                        "analyze", "analyse", "review", "describe", "tell me",
                        "answer", "extract", "find", "list", "identify",
                        "according to", "based on", "from the document", "from this",
                        "in the document", "the document says", "does it",
                    )
                    _prompt_lower = user_prompt.lower()
                    _is_qa_mode   = any(kw in _prompt_lower for kw in _qa_keywords)

                    if _is_qa_mode:
                        _doc_block = (
                            f"\n\nDOCUMENT TO ANALYSE (the user's question is about this):\n"
                            f"The following is the full text of a document provided by the user.\n"
                            f"Your response MUST be grounded in this document's content.\n"
                            f"Quote or cite specific parts when relevant.\n"
                            f"If the document does not contain information needed to answer "
                            f"the question, say so explicitly rather than guessing.\n"
                            f"{'[NOTE: Document was truncated to first 40,000 chars]' if _truncated else ''}\n"
                            f"---\n{_doc_preview}\n---\n"
                        )
                    else:
                        _doc_block = (
                            f"\n\nREFERENCE DOCUMENT (treat as source material):\n"
                            f"The following is the full text of a document provided by the user.\n"
                            f"You MUST use its content directly where the rules require it.\n"
                            f"Do NOT paraphrase, fabricate, or substitute this content.\n"
                            f"Cite or reference specific sections where relevant.\n"
                            f"{'[NOTE: Document was truncated to first 40,000 chars]' if _truncated else ''}\n"
                            f"---\n{_doc_preview}\n---\n"
                        )
                else:
                    _doc_block = ""

                # Build citation instruction if we have sources with URLs
                _citation_block = ""
                if _src_ref_lines:
                    _citation_block = (
                        f"\n\nSOURCE CITATION REQUIREMENTS:\n"
                        f"The following sources were retrieved for this query. "
                        f"You MUST cite them inline in your response using [Source N] notation "
                        f"wherever you use information from them.\n"
                        f"At the end of your response, include a 'References' section that "
                        f"lists each source you cited with its full URL.\n"
                        f"Sources available:\n"
                        + "\n".join(_src_ref_lines)
                        + "\n"
                    )

                gen_prompt = (
                    f"You are generating content for a user request. "
                    f"You MUST satisfy every constraint below.\n\n"
                    f"USER REQUEST:\n{user_prompt}\n\n"
                    f"{constraint_block}\n"
                    f"{_citation_block}"
                    f"{violation_feedback}"
                    f"{_doc_block}\n\n"
                    f"ADDITIONAL CONTEXT FROM RESEARCH:\n"
                    + ("\n".join(ctx_parts) if ctx_parts else "None")
                    + "\n\nGenerate a detailed, helpful response that explicitly states "
                      "all relevant values as numbers and uses the exact variable labels "
                      "specified in the constraints above.\n\n"
                      "FORMATTING RULES — READ CAREFULLY:\n"
                      "1. Do NOT start your response with a block of constraint flag lines "
                      "(lines like `variable_name: true` or `variable_name: false`). "
                      "ALL such verification label lines must go at the very END of your "
                      "response, after all prose and explanatory content, under the exact "
                      "section header: 'CONSTRAINT VERIFICATION LABELS'.\n"
                      "2. Every numbered list item you start (1. 2. 3. ...) MUST be "
                      "completed with full content. Never leave a trailing number with "
                      "no text after it.\n"
                      "3. Write in natural prose — constraint labels exist for machine "
                      "verification only and must never interrupt or dominate your content."
                )

                try:
                    _model_display = llm_config.get("model", "model")
                    _doc_note = f" (+ {len(_doc_text):,} char document)" if _doc_text else ""
                    status.info(f"⚡ {iter_label} — generating with {_model_display}{_doc_note}…")
                    stream_box = st.empty()
                    collected  = []

                    if llm_config.get("provider") == "anthropic":
                        import anthropic as _ant
                        import hashlib as _hl
                        _ant_ck = _hl.md5(("anthropic" + llm_config["model"] + api_key).encode()).hexdigest()
                        if _ant_ck not in m2._CLIENT_CACHE:
                            m2._CLIENT_CACHE[_ant_ck] = _ant.Anthropic(api_key=api_key)
                        _anth_client = m2._CLIENT_CACHE[_ant_ck]
                        with _anth_client.messages.stream(
                            model=llm_config["model"], max_tokens=16000,
                            messages=[{"role":"user","content":gen_prompt}]
                        ) as stream:
                            for text_chunk in stream.text_stream:
                                collected.append(text_chunk)
                                live = "".join(collected)
                                safe = live.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                                stream_box.markdown(
                                    f'<div class="gen-output" style="max-height:200px">{safe}</div>',
                                    unsafe_allow_html=True)

                    elif llm_config.get("provider") == "openai":
                        import openai as _oai
                        _oai_client = _oai.OpenAI(api_key=api_key)
                        _oai_model  = llm_config["model"]
                        # o-series and gpt-5.x use max_completion_tokens, not max_tokens
                        _uses_completion_tokens = (
                            _oai_model in {"o3","o3-pro","o3-mini","o4-mini","o1","o1-mini","o1-pro"}
                            or _oai_model.startswith("gpt-5")
                        )
                        _oai_kwargs = {
                            "model"   : _oai_model,
                            "stream"  : True,
                            "messages": [{"role":"user","content":gen_prompt}],
                        }
                        if _uses_completion_tokens:
                            _oai_kwargs["max_completion_tokens"] = 16000
                        else:
                            _oai_kwargs["max_tokens"] = 16000
                        stream = _oai_client.chat.completions.create(**_oai_kwargs)
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                collected.append(delta)
                                live = "".join(collected)
                                safe = live.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                                stream_box.markdown(
                                    f'<div class="gen-output" style="max-height:200px">{safe}</div>',
                                    unsafe_allow_html=True)

                    else:  # Google Gemini — no streaming SDK, single call
                        stream_box.info(f"Generating with {llm_config.get('model','Gemini')}...")
                        result = m2._call_llm(gen_prompt, llm_config, max_tokens=16000)
                        collected = [result]
                        safe = result.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        stream_box.markdown(
                            f'<div class="gen-output" style="max-height:200px">{safe}</div>',
                            unsafe_allow_html=True)

                    draft_text = "".join(collected)
                    stream_box.empty()
                    status.empty()
                    if not draft_text.strip():
                        st.error(f"{llm_config.get('model','LLM')} returned an empty response. Check your API key and model access."); st.stop()
                except Exception as e:
                    st.error(f"Draft generation failed: {e}"); st.stop()

            # ── Audit ─────────────────────────────────────────────────────────
            if structured_rules and draft_text:
                prog.progress(min(99, 55 + attempt * 6), text=f"🔍 {iter_label} -- auditing...")
                status.info(f"M2 auditing {len(structured_rules)} rule(s)…")
                try:
                    audit_results = m2.structured_audit(draft_text, structured_rules, api_key, llm_config=llm_config)
                except Exception as e:
                    st.error(f"Audit failed: {e}"); st.stop()
                status.empty()
            else:
                audit_results = []

            # ── LTN ───────────────────────────────────────────────────────────
            ltn_score  = 0.0
            violations = []
            if has_ltn and audit_results:
                try:
                    ltn_score, violations = m1.verify_and_report(audit_results)
                except Exception as e:
                    st.warning(f"LTN scoring failed: {e}")

            # Store Qdrant audit records
            if has_m3 and qdrant_client and audit_results:
                try:
                    for ar in audit_results:
                        m3.store_audit_result(qdrant_client, ar, run_id=run_id)
                    brain_records["audit"] = m3.get_all_records(
                        qdrant_client, "audit", run_id=run_id)
                except Exception:
                    pass

            # Save this iteration
            iter_record = {
                "attempt"   : attempt,
                "draft"     : draft_text,
                "audit"     : audit_results,
                "ltn_score" : ltn_score,
                "violations": violations,
            }
            if len(iteration_history) > 0:
                iter_record["diff"] = _diff_sentences(
                    iteration_history[-1]["draft"], draft_text)
            iteration_history.append(iter_record)

            # Always keep the latest for halt message
            final_audit_results = audit_results
            final_ltn_score     = ltn_score
            final_draft         = draft_text

            passed_count = sum(1 for r in audit_results if r.get("satisfies"))
            total_count  = len(audit_results)
            status.info(f"{'✅' if ltn_score >= LTN_THRESHOLD else '⚠️'} {iter_label} "
                        f"— LTN: {ltn_score:.4f} | {passed_count}/{total_count} rules passed")

            if (not audit_results) or ltn_score >= LTN_THRESHOLD:
                passed = True
                break

            # Build violation feedback for the next iteration
            if attempt < MAX_ITER:
                violation_feedback = _build_violation_feedback(violations)
                time.sleep(0.5)

        # Done
        prog.progress(100, text="✅ Done!")
        time.sleep(0.3)
        prog.empty()
        status.empty()

        results.update({
            "draft"            : final_draft,
            "audit"            : final_audit_results,
            "ltn_score"        : final_ltn_score,
            "violations"       : [r for r in final_audit_results if not r.get("satisfies")],
            "passed"           : passed,
            "iterations_used"  : attempt,
            "max_iterations"   : MAX_ITER,
            "ltn_threshold"    : LTN_THRESHOLD,
            "brain_records"    : brain_records,
            "run_id"           : run_id,
            "iteration_history": iteration_history,
            "user_prompt"      : user_prompt.strip(),
            "llm_provider"     : _prov["id"],
            "llm_model"        : _model_choice,
            "llm_provider_name": _provider_name,
        })
        st.session_state.results           = results
        st.session_state.brain_records      = brain_records
        st.session_state['_pipeline_running'] = False
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS TABS
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.results:
        res  = st.session_state.results

        # Show contradiction warnings at the top if any were found
        _ctrs = res.get("contradictions", [])
        if _ctrs:
            for _ctr in _ctrs:
                st.warning(
                    f"⚠️ **Contradicting rules detected:** {_ctr['reason']}  \n"
                    f"Rule A: *{_ctr['rule_a'].get('display','')}*  ·  "
                    f"Rule B: *{_ctr['rule_b'].get('display','')}*  \n"
                    f"These rules cannot both be satisfied — consider revising them."
                )

        # Build a short topic slug for filenames from the user prompt
        def _make_slug(text: str, max_len: int = 40) -> str:
            import re as _re
            slug = _re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
            slug = slug.strip("_")[:max_len].rstrip("_")
            return slug if slug else "output"

        _topic_slug = _make_slug(res.get("user_prompt", "output"))

        tabs = st.tabs(["📊 Verdict", "📄 Draft", "🔍 Audit",
                        "💡 Insight", "🧠 Second Brain", "🌐 Sources", "🗂 Raw JSON", "📋 Report"])

        # ── Tab 1: Verdict ────────────────────────────────────────────────────
        with tabs[0]:
            audit  = res.get("audit", [])
            score  = res.get("ltn_score")
            thresh = res.get("ltn_threshold", 0.80)
            passed = res.get("passed", False)
            iters  = res.get("iterations_used", 1)
            max_it = res.get("max_iterations", 5)
            viols  = res.get("violations", [])
            passed_count = sum(1 for r in audit if r.get("satisfies"))
            failed_count = len(audit) - passed_count

            if score is not None:
                sc = "score-pass" if score >= thresh else ("score-warn" if score >= 0.5 else "score-fail")
                vt = "VERIFIED ✓" if passed else ("MARGINAL ⚠" if score >= 0.5 else "FAILED ✗")
                vc = "#6dcea8"   if passed else ("#e8c06d" if score >= 0.5 else "#e8736d")
                st.markdown(f"""
                <div class="glass-panel" style="text-align:center;padding:1.6rem;">
                    <div class="score-big {sc}">{score:.4f}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:{vc};font-weight:600;letter-spacing:0.1em;margin-top:0.2rem;">{vt}</div>
                    <div class="score-label" style="margin-top:0.3rem;">LTN Universal Verification Score (threshold: {thresh:.2f})</div>
                    <div style="font-size:0.68rem;color:rgba(200,169,110,0.6);margin-top:0.4rem;font-family:'DM Mono',monospace;">Generated by {res.get('llm_provider_name','Claude')} · {res.get('llm_model','')}</div>
                </div>""", unsafe_allow_html=True)

            if audit:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Checked",   len(audit))
                c2.metric("Passed ✅", passed_count)
                c3.metric("Failed ❌", failed_count)
                c4.metric("Iterations", f"{iters}/{max_it}")

            if not passed and viols:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">⚠ Remaining Violations</div>', unsafe_allow_html=True)
                for v in viols:
                    e_rule = v.get('rule_display','').replace("&","&amp;").replace("<","&lt;")
                    e_expl = v.get('explanation','').replace("&","&amp;").replace("<","&lt;")
                    score_v = v.get('compliance_score', 0)
                    st.markdown(f"""
                    <div class="audit-row">
                        <div class="audit-sidebar fail"></div>
                        <div class="audit-body">
                            <div class="audit-top">
                                <span class="audit-badge fail">FAIL {score_v:.2f}</span>
                                <span class="audit-rule-text">{e_rule}</span>
                            </div>
                            <div class="audit-explanation">{e_expl}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            elif passed:
                st.success(f"✅ All constraints satisfied after {iters} iteration(s).")

        # ── Tab 2: Draft ──────────────────────────────────────────────────────
        with tabs[1]:
            draft = res.get("draft", "")
            if draft:
                st.markdown('<div class="section-label">✍ Final Verified Draft</div>', unsafe_allow_html=True)
                safe = draft.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'<div class="gen-output">{safe}</div>', unsafe_allow_html=True)
                # ── Download buttons — always show TXT, show PDF if generation succeeds ──
                dl_col1, dl_col2 = st.columns(2)

                # TXT is always available — no dependencies, never fails
                with dl_col1:
                    st.download_button(
                        "⬇ Download as TXT",
                        data      = draft,
                        file_name = f"{_topic_slug}_{res.get('run_id','output')}.txt",
                        mime      = "text/plain",
                        key       = "dl_draft_txt",
                        use_container_width = True,
                    )

                # PDF — attempt generation, show button if successful
                with dl_col2:
                    try:
                        _pdf_bytes = _draft_to_pdf(
                            draft,
                            run_id          = res.get("run_id", ""),
                            ltn_score       = res.get("ltn_score"),
                            rules_passed    = sum(1 for r in res.get("audit",[]) if r.get("satisfies")),
                            rules_total     = len(res.get("audit",[])),
                            iterations_used = res.get("iterations_used", 1),
                            llm_label       = f"{res.get('llm_provider_name','Claude')} / {res.get('llm_model','')}",
                        )
                        st.download_button(
                            "⬇ Download as PDF",
                            data      = _pdf_bytes,
                            file_name = f"{_topic_slug}_{res.get('run_id','output')}.pdf",
                            mime      = "application/pdf",
                            key       = "dl_draft_pdf",
                            use_container_width = True,
                        )
                    except Exception as _pdf_err:
                        st.button(
                            "⬇ PDF unavailable",
                            disabled = True,
                            key      = "dl_draft_pdf_disabled",
                            help     = f"PDF generation failed: {_pdf_err}",
                            use_container_width = True,
                        )
            else:
                st.info("No draft available.")

        # ── Tab 3: Audit Detail ───────────────────────────────────────────────
        with tabs[2]:
            audit = res.get("audit", [])
            if not audit:
                st.info("No audit results. Add rules and re-run.")
            else:
                st.markdown('<div class="section-label">🔍 Per-Rule Results</div>', unsafe_allow_html=True)
                pass_pct = int(100 * passed_count / len(audit)) if audit else 0
                st.markdown(f"""
                <div style="margin-bottom:1.2rem;">
                  <div style="display:flex;justify-content:space-between;font-size:0.68rem;font-family:'DM Mono',monospace;color:rgba(232,228,220,0.4);margin-bottom:0.35rem;">
                    <span>{passed_count} passed</span><span>{failed_count} failed</span>
                  </div>
                  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{pass_pct}%;background:linear-gradient(90deg,#6dcea8,#4ab898);border-radius:3px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                for r in audit:
                    compliance = r.get("compliance_score", 1.0 if r.get("satisfies") else 0.0)
                    if compliance >= 0.8:
                        s_cls, badge_lbl = "pass", f"PASS {compliance:.2f}"
                    elif compliance >= 0.5:
                        s_cls, badge_lbl = "warn", f"PARTIAL {compliance:.2f}"
                    else:
                        s_cls, badge_lbl = "fail", f"FAIL {compliance:.2f}"

                    rid       = r.get("rule_id",""); ridlbl = (rid+1) if isinstance(rid,int) else rid
                    sym_tag   = '<span class="audit-sym-tag">symbolic ✓</span>' if r.get("symbolic_check_used") else ""
                    dw        = r.get("domain_warning","")
                    dw_html   = f'<div class="audit-domain-warn">⚠ {dw}</div>' if dw else ""
                    p_conf    = r.get("premise_confidence", 1.0)
                    c_conf    = r.get("conclusion_confidence", compliance)
                    c_color   = "#6dcea8" if c_conf >= 0.5 else "#e8736d"
                    rule_disp = r.get("rule_display","").replace("&","&amp;").replace("<","&lt;")
                    extr_raw  = str(r.get("extracted_value_raw","N/A")).replace("<","&lt;")
                    extr_num  = r.get("extracted_value_num")
                    extr_ns   = f"{extr_num:.4g}" if extr_num is not None else "—"
                    unit_note = r.get("unit_conversion_note","—").replace("<","&lt;")
                    scope_val = r.get("scope","—").upper()
                    expl      = r.get("explanation","No explanation.").replace("<","&lt;")
                    src_name  = r.get("source_name","")
                    src_note  = f'<span style="font-size:0.62rem;color:rgba(200,169,110,0.5);margin-left:auto;">{src_name}</span>' if src_name else ""

                    # ── Audit card: header via HTML (reliable), body via native Streamlit ──
                    # Reason: Streamlit's markdown parser chokes on deeply nested HTML
                    # in a single call — outer renders but inner shows as raw text.
                    verdict_icon = "✅" if compliance >= 0.8 else ("⚠️" if compliance >= 0.5 else "❌")
                    method_label = "Symbolic ✓" if r.get("symbolic_check_used") else "Semantic"
                    border_color = "#6dcea8" if compliance >= 0.8 else ("#e8c06d" if compliance >= 0.5 else "#e8736d")
                    src_label    = f" · {src_name}" if src_name else ""

                    # Card wrapper — one shallow div, reliably rendered
                    st.markdown(
                        f'<div style="border-left:4px solid {border_color};border-radius:0 12px 12px 0;'
                        f'background:rgba(255,255,255,0.025);padding:0.85rem 1.1rem;margin-bottom:0.75rem;">',
                        unsafe_allow_html=True
                    )

                    # Rule header line
                    st.markdown(
                        f'{verdict_icon} **R{ridlbl} — {rule_disp}**  '
                        f'`{badge_lbl}` · {method_label}{src_label}'
                    )

                    # Domain warning if present
                    if dw:
                        st.warning(f"⚠ Domain issue: {dw}")

                    # Pills as native metric columns — always renders correctly
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Extracted", extr_raw[:22] if extr_raw not in ("N/A","") else "—")
                    pc2.metric("Numeric", extr_ns)
                    pc3.metric("Scope", scope_val)
                    pc4.metric("Unit conv.", unit_note[:18] if unit_note not in ("—","none","") else "—")
                    pc5.metric("Score", f"{compliance:.2f}")

                    # Compliance progress bar — native, always works
                    bar_pct = max(0.01, compliance)  # st.progress needs > 0
                    st.progress(bar_pct, text=f"Compliance: {compliance:.0%}")

                    # Explanation
                    st.caption(f"💬 {expl}")

                    # Close card div
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("🗂 Structured Rule JSON"):
                    for sr in res.get("structured_rules", []):
                        st.json(sr)

        # ── Tab 4: INSIGHT PANEL ──────────────────────────────────────────────
        with tabs[3]:
            st.markdown('<div class="section-label">💡 How The Response Was Generated</div>', unsafe_allow_html=True)
            history = res.get("iteration_history", [])
            sr_list = res.get("structured_rules", [])

            # ── 4a. Rules enforced (with source badges) ───────────────────────
            st.markdown("**Rules enforced in this run:**")
            for i, r in enumerate(sr_list):
                sn    = r.get("source_name","User")
                sc    = _insight_sc(sn)
                disp  = r.get("display", r.get("original",""))[:80]
                op    = r.get("operator","")
                th    = r.get("threshold")
                unit  = r.get("unit","")
                scope = r.get("scope","always").upper()
                th_str = f"{op} {th} {unit}".strip() if th is not None else op
                disp_e = disp.replace("&","&amp;").replace("<","&lt;")
                st.markdown(f"""
                <div class="insight-rule-row">
                  <span class="rule-num">R{i+1}</span>
                  <span style="flex:1;font-family:'DM Mono',monospace;font-size:0.78rem;">{disp_e}</span>
                  <span style="font-size:0.62rem;color:rgba(232,228,220,0.35);font-family:'DM Mono',monospace;">{scope}</span>
                  <span style="font-size:0.62rem;color:rgba(200,169,110,0.6);font-family:'DM Mono',monospace;margin:0 0.4rem;">{th_str}</span>
                  <span class="src-badge {sc}">{sn.upper()[:4]}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 4b. LTN score progression ─────────────────────────────────────
            if history:
                st.markdown("**LTN score progression across iterations:**")
                score_html = '<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem;">'
                for h in history:
                    sc_v  = h["ltn_score"]
                    sc_cls = "iter-score-pass" if sc_v >= res.get("ltn_threshold",0.8) else ("iter-score-warn" if sc_v >= 0.5 else "iter-score-fail")
                    n_pass = sum(1 for r in h["audit"] if r.get("satisfies"))
                    n_tot  = len(h["audit"])
                    score_html += f'<span class="iter-badge">iter {h["attempt"]}</span><span class="{sc_cls}">{sc_v:.4f}</span><span style="font-size:0.72rem;color:rgba(232,228,220,0.3);margin-right:0.6rem;">({n_pass}/{n_tot})</span>'
                    if h["attempt"] < len(history):
                        score_html += '<span style="color:rgba(200,169,110,0.3);font-size:0.8rem;">→</span>'
                score_html += '</div>'
                st.markdown(score_html, unsafe_allow_html=True)

            # ── 4c. Per-iteration breakdown ───────────────────────────────────
            st.markdown("**What changed between iterations:**")
            for h in history:
                viols_h = h.get("violations", [])
                diff_h  = h.get("diff", {})
                n_pass  = sum(1 for r in h["audit"] if r.get("satisfies"))
                n_tot   = len(h["audit"])
                lsc     = h["ltn_score"]
                sc_col  = "#6dcea8" if lsc >= res.get("ltn_threshold",0.8) else ("#e8c06d" if lsc >= 0.5 else "#e8736d")

                with st.expander(f"Iteration {h['attempt']} — LTN: {lsc:.4f} | {n_pass}/{n_tot} rules passed", expanded=(h['attempt']==len(history))):
                    if viols_h:
                        st.markdown("**Rules that failed this iteration:**")
                        for v in viols_h:
                            vd = v.get('rule_display','').replace("<","&lt;")
                            ve = v.get('explanation','').replace("<","&lt;")
                            vs = v.get('compliance_score',0)
                            st.markdown(f"""
                            <div style="padding:0.5rem 0.75rem;background:rgba(232,115,109,0.07);border-left:3px solid #e8736d;border-radius:0 8px 8px 0;margin-bottom:0.4rem;font-size:0.79rem;">
                              <span style="color:#e8736d;font-weight:600;">✗ {vd}</span>
                              <span style="color:rgba(232,228,220,0.4);font-size:0.72rem;margin-left:0.5rem;">score: {vs:.2f}</span>
                              <div style="color:rgba(232,228,220,0.5);margin-top:0.25rem;">{ve}</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color:#6dcea8;font-size:0.82rem;padding:0.4rem;">✓ All rules satisfied this iteration.</div>', unsafe_allow_html=True)

                    if diff_h and (diff_h.get("added") or diff_h.get("removed")):
                        st.markdown("**Changes from previous draft:**")
                        for line in diff_h.get("removed",[]):
                            safe = line.replace("&","&amp;").replace("<","&lt;")
                            if safe.strip():
                                st.markdown(f'<div class="diff-remove">− {safe}</div>', unsafe_allow_html=True)
                        for line in diff_h.get("added",[]):
                            safe = line.replace("&","&amp;").replace("<","&lt;")
                            if safe.strip():
                                st.markdown(f'<div class="diff-add">+ {safe}</div>', unsafe_allow_html=True)
                    elif h["attempt"] == 1:
                        st.caption("First iteration — no previous draft to compare.")

                    # Show this iteration's draft snippet
                    draft_snippet = h["draft"][:600] + ("…" if len(h["draft"])>600 else "")
                    safe_snip = draft_snippet.replace("&","&amp;").replace("<","&lt;")
                    st.markdown(f'<div class="gen-output" style="max-height:180px;font-size:0.78rem;">{safe_snip}</div>', unsafe_allow_html=True)

            # ── 4d. Final confidence score explanation ────────────────────────
            if res.get("ltn_score") is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                final_sc = res["ltn_score"]
                thresh_v = res.get("ltn_threshold", 0.8)
                st.markdown("**Final confidence score explained:**")
                st.markdown(f"""
                <div class="glass-panel">
                  <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:{'#6dcea8' if final_sc>=thresh_v else '#e8736d'};">{final_sc:.4f}</div>
                  <div style="font-size:0.8rem;color:rgba(232,228,220,0.6);margin-top:0.5rem;line-height:1.65;">
                    This is the LTN (Logic Tensor Network) universal verification score. It aggregates how
                    well every rule was satisfied using fuzzy Reichenbach implication:
                    <code style="font-family:'DM Mono',monospace;font-size:0.75rem;">∀ rule: Premise(rule) → Conclusion(rule)</code>
                    combined with pMeanError aggregation (p=2). Each rule contributes a compliance score
                    between 0 and 1 — a rule 80% satisfied contributes 0.8, not binary 0 or 1.
                    A score above {thresh_v:.2f} means all constraints are sufficiently satisfied.
                    Symbolic rules (numerical) are hard-verified with Python math. Semantic rules
                    (boolean/qualitative) are verified by Claude acting as a strict auditor.
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Tab 5: Second Brain ───────────────────────────────────────────────
        with tabs[4]:
            brain = res.get("brain_records", st.session_state.brain_records or {})
            st.markdown("""
            <div style="margin-bottom:1.2rem;">
                <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#f0ebe0;">🧠 Second Brain</div>
                <div style="font-size:0.72rem;color:rgba(232,228,220,0.4);text-transform:uppercase;letter-spacing:0.1em;">Qdrant symbolic memory</div>
            </div>""", unsafe_allow_html=True)

            rules_recs  = brain.get("rules",   [])
            source_recs = brain.get("sources", [])
            audit_recs  = brain.get("audit",   [])
            total       = len(rules_recs) + len(source_recs) + len(audit_recs)

            sc1,sc2,sc3,sc4 = st.columns(4)
            with sc1: st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{total}</div><div class="brain-stat-label">Total Nodes</div></div>', unsafe_allow_html=True)
            with sc2: st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(rules_recs)}</div><div class="brain-stat-label">Rule Nodes</div></div>', unsafe_allow_html=True)
            with sc3: st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(source_recs)}</div><div class="brain-stat-label">Source Nodes</div></div>', unsafe_allow_html=True)
            with sc4: st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(audit_recs)}</div><div class="brain-stat-label">Audit Nodes</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if total == 0:
                st.info("No symbolic references stored yet — run the pipeline first.")
            else:
                brain_tab = st.radio("Collection", ["Rules", "Sources", "Audit"],
                                     horizontal=True, key="brain_tab_select",
                                     label_visibility="collapsed")
                records = {"Rules":rules_recs,"Sources":source_recs,"Audit":audit_recs}[brain_tab]
                if not records:
                    st.info(f"No {brain_tab.lower()} records in this run.")
                else:
                    st.markdown(f'<div class="section-label">📦 {brain_tab} — {len(records)} node(s)</div>', unsafe_allow_html=True)
                    for rec in records:
                        rtype     = rec.get("record_type","")
                        text      = rec.get("text","")[:110]
                        stored_at = rec.get("stored_at","")[:19].replace("T"," ")
                        if rtype=="rule":
                            nature    = rec.get("rule_nature","constraint")
                            badge_cls = f"sym-type-{nature}"; badge_lbl = nature.upper()
                        elif rtype=="source":
                            badge_cls = "sym-type-source"; badge_lbl = "SOURCE"
                        elif rtype=="audit":
                            ok = rec.get("satisfies",False)
                            badge_cls = "sym-type-audit-pass" if ok else "sym-type-audit-fail"
                            badge_lbl = "PASS" if ok else "FAIL"
                        else:
                            badge_cls = "sym-type-constraint"; badge_lbl = rtype.upper()
                        chips_html = ""
                        if rtype=="rule":
                            if rec.get("variable"):  chips_html += f'<span class="sym-meta-chip highlight">{rec["variable"]}</span>'
                            if rec.get("operator"):  chips_html += f'<span class="sym-meta-chip highlight">{rec["operator"]}</span>'
                            if rec.get("threshold") is not None: chips_html += f'<span class="sym-meta-chip highlight">{rec["threshold"]} {rec.get("unit","")}</span>'
                            if rec.get("scope"):     chips_html += f'<span class="sym-meta-chip">scope:{rec["scope"]}</span>'
                            if rec.get("source_name"): chips_html += f'<span class="sym-meta-chip">{rec["source_name"]}</span>'
                        elif rtype=="audit":
                            cs = rec.get("compliance_score", rec.get("conclusion_confidence",0))
                            chips_html += f'<span class="sym-meta-chip highlight">score:{cs:.2f}</span>'
                            if rec.get("symbolic_check_used"): chips_html += '<span class="sym-meta-chip highlight">symbolic✓</span>'
                        safe_text = text.replace("&","&amp;").replace("<","&lt;")
                        st.markdown(f"""
                        <div class="sym-node">
                          <div class="sym-node-top">
                            <span class="sym-type-badge {badge_cls}">{badge_lbl}</span>
                            <span class="sym-node-text">{safe_text}</span>
                            <span class="sym-ts">{stored_at}</span>
                          </div>
                          <div class="sym-meta-row">{chips_html}</div>
                          <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:rgba(200,169,110,0.35);margin-top:0.3rem;">· · · [384-dim cosine embedding · sentence-transformers/all-MiniLM-L6-v2] · · ·</div>
                        </div>""", unsafe_allow_html=True)

        # ── Tab 6: Sources ────────────────────────────────────────────────────
        with tabs[5]:
            sources = [s for s in res.get("sources", [])
                       if s.get("context","").strip()
                       and len(s.get("context","").strip()) >= 50
                       and (s.get("reference","") or "").strip().lower() not in ("","none")]
            if sources:
                st.markdown('<div class="section-label">🌐 Research Sources Used</div>', unsafe_allow_html=True)
                st.caption(f"{len(sources)} source(s) fetched during this run — click any URL to open the original page.")
                for _si, src in enumerate(sources):
                    _sn    = src.get("source_name", "Source")
                    _title = src.get("title", "")
                    _ref   = (src.get("reference","") or src.get("source","") or "").strip()
                    _ctx   = src.get("context","")
                    _badge = _src_badge(_sn)
                    with st.expander(f"📖 [{_si+1}] {_sn} — {_title}", expanded=False):
                        # Context body — same style as original
                        if _ctx:
                            st.write(_ctx[:600] + ("…" if len(_ctx) > 600 else ""))
                        # Full URL as a clickable pill — original style, full URL not truncated
                        if _ref and _ref not in ("None","none",""):
                            st.markdown(
                                f'<a class="ref-pill" href="{_ref}" target="_blank" '
                                f'style="word-break:break-all;">🔗 {_ref}</a>',
                                unsafe_allow_html=True,
                            )
            else:
                st.info("No research sources in this run.")

        # ── Tab 7: Raw JSON ───────────────────────────────────────────────────
        with tabs[6]:
            st.markdown('<div class="section-label">🗂 Full Pipeline Output</div>', unsafe_allow_html=True)
            display = {k: v for k, v in res.items()
                       if k not in ("draft","brain_records","iteration_history")}
            display["draft_preview"] = (res.get("draft","")[:500]+"…") if res.get("draft") else ""
            display["iterations_summary"] = [
                {"attempt":h["attempt"],"ltn_score":h["ltn_score"],
                 "rules_passed":sum(1 for r in h["audit"] if r.get("satisfies")),
                 "rules_total":len(h["audit"])}
                for h in res.get("iteration_history",[])
            ]
            st.json(display)
            st.download_button("⬇ Download JSON",
                               data=json.dumps(res, indent=2, default=str),
                               file_name="pipeline_results.json",
                               mime="application/json", key="dl_json")

        # ── Tab 8: Human-Readable Transparency Report ───────────────────────
        with tabs[7]:
            st.markdown('<div class="section-label">📋 Transparency Report</div>', unsafe_allow_html=True)

            _audit   = res.get("audit", [])
            _score   = res.get("ltn_score", 0)
            _thresh  = res.get("ltn_threshold", 0.8)
            _passed  = res.get("passed", False)
            _iters   = res.get("iterations_used", 1)
            _max_it  = res.get("max_iterations", 5)
            _run_id  = res.get("run_id", "")
            _rules   = res.get("structured_rules", [])
            _history = res.get("iteration_history", [])
            _sources = res.get("sources", [])
            _n_pass  = sum(1 for r in _audit if r.get("satisfies"))
            _n_fail  = len(_audit) - _n_pass
            _verdict = "PASSED" if _passed else "DID NOT PASS"
            _v_color = "#6dcea8" if _passed else "#e8736d"

            # ── Section 1: Plain-English Summary ─────────────────────────────
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
     border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1.2rem;">
  <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#f0ebe0;margin-bottom:0.6rem;">
    What happened in this run?
  </div>
  <p style="font-size:0.88rem;color:rgba(232,228,220,0.7);line-height:1.75;margin:0;">
    You asked the system to generate content and verify it against <strong style="color:#c8a96e">{len(_rules)} rules</strong>.
    The system {"researched the topic online and " if _sources else ""}parsed your rules into formal logical constraints,
    generated a draft, then mathematically checked every rule against the output.
    After <strong style="color:#c8a96e">{_iters} iteration{"s" if _iters > 1 else ""}</strong> of generate → verify → rewrite,
    the final LTN verification score was
    <strong style="color:{_v_color}">{_score:.4f}</strong> against a pass threshold of {_thresh:.2f}
    — the run <strong style="color:{_v_color}">{_verdict}</strong>.
    {f"<br><br><em style='color:rgba(232,228,220,0.45);font-size:0.82rem;'>The system rewrote the draft {_iters-1} time(s) based on specific violation feedback before reaching this result.</em>" if _iters > 1 else ""}
  </p>
</div>""", unsafe_allow_html=True)

            # ── Section 2: What is the LTN Score? ────────────────────────────
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
     border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
  <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
              color:#c8a96e;font-weight:600;margin-bottom:0.7rem;">
    What does the score {_score:.4f} mean?
  </div>
  <p style="font-size:0.84rem;color:rgba(232,228,220,0.65);line-height:1.75;margin:0 0 0.6rem 0;">
    The LTN (Logic Tensor Network) score is a single number between 0 and 1 that summarises
    how well the generated content satisfied <em>all</em> your rules simultaneously.
    It is not an average — it uses fuzzy logic so that a single badly-violated rule
    pulls the whole score down significantly.
  </p>
  <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.6rem;">
    <div style="font-size:0.81rem;color:rgba(232,228,220,0.5);">
      <span style="color:#6dcea8;font-weight:600;">≥ {_thresh:.2f}</span> — PASS (all constraints sufficiently satisfied)
    </div>
    <div style="font-size:0.81rem;color:rgba(232,228,220,0.5);">
      <span style="color:#e8c06d;font-weight:600;">0.50 – {_thresh:.2f}</span> — MARGINAL (minor violations)
    </div>
    <div style="font-size:0.81rem;color:rgba(232,228,220,0.5);">
      <span style="color:#e8736d;font-weight:600;">below 0.50</span> — FAIL (significant violations)
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Section 3: Rules — plain English per rule ─────────────────────
            st.markdown("""
<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
            color:#c8a96e;font-weight:600;margin-bottom:0.8rem;">
  How did each rule perform?
</div>""", unsafe_allow_html=True)

            for r in _audit:
                compliance  = r.get("compliance_score", 1.0 if r.get("satisfies") else 0.0)
                satisfies   = r.get("satisfies", False)
                rule_disp   = r.get("rule_display","").replace("<","&lt;")
                explanation = r.get("explanation","").replace("<","&lt;")
                extracted   = str(r.get("extracted_value_raw","N/A")).replace("<","&lt;")
                scope_val   = r.get("scope","always")
                src_name    = r.get("source_name","")
                method      = "hard math (symbolic)" if r.get("symbolic_check_used") else "AI semantic judgement"
                conv_note   = r.get("unit_conversion_note","")

                if compliance >= 0.8:
                    icon = "✅"; bar_c = "#6dcea8"; verdict_txt = f"Passed ({int(compliance*100)}% compliance)"
                elif compliance >= 0.5:
                    icon = "⚠️"; bar_c = "#e8c06d"; verdict_txt = f"Partially satisfied ({int(compliance*100)}% compliance)"
                else:
                    icon = "❌"; bar_c = "#e8736d"; verdict_txt = f"Failed ({int(compliance*100)}% compliance)"

                scope_plain = {
                    "always"      : "checked every occurrence",
                    "initial"     : "checked only the first/starting value",
                    "final"       : "checked only the final value",
                    "maximum"     : "checked the highest value found",
                    "minimum"     : "checked the lowest value found",
                    "conditional" : "checked only when a specific condition was met",
                    "context_only": "verified the value was used as context (not enforced strictly)",
                }.get(scope_val, scope_val)

                src_note = f" · Source: {src_name}" if src_name else ""
                conv_html = f'<div style="font-size:0.76rem;color:rgba(200,169,110,0.55);margin-top:0.2rem;">Unit conversion: {conv_note}</div>' if conv_note and conv_note.lower() not in ("none","") else ""

                st.markdown(f"""
<div style="border-left:4px solid {bar_c};border-radius:0 12px 12px 0;
     background:rgba(255,255,255,0.02);padding:0.9rem 1.2rem;margin-bottom:0.75rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;flex-wrap:wrap;">
    <span style="font-size:1rem;">{icon}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#e8e4dc;flex:1;">
      {rule_disp}
    </span>
    <span style="font-size:0.68rem;font-family:'DM Mono',monospace;
                 color:{bar_c};background:rgba(255,255,255,0.04);
                 border-radius:5px;padding:0.1rem 0.4rem;white-space:nowrap;">
      {verdict_txt}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem 1.5rem;
              font-size:0.79rem;color:rgba(232,228,220,0.55);margin-bottom:0.5rem;">
    <div><strong style="color:rgba(232,228,220,0.35);font-size:0.68rem;
                        text-transform:uppercase;letter-spacing:0.08em;">
      What the system found
    </strong><br>{extracted[:120] + ("…" if len(extracted)>120 else "")}</div>
    <div><strong style="color:rgba(232,228,220,0.35);font-size:0.68rem;
                        text-transform:uppercase;letter-spacing:0.08em;">
      How it was checked
    </strong><br>{method} · {scope_plain}{src_note}</div>
  </div>
  {conv_html}
  <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin:0.4rem 0;">
    <div style="height:100%;width:{int(compliance*100)}%;background:{bar_c};border-radius:2px;"></div>
  </div>
  <div style="font-size:0.79rem;color:rgba(232,228,220,0.45);margin-top:0.35rem;line-height:1.55;">
    {explanation}
  </div>
</div>""", unsafe_allow_html=True)

            # ── Section 4: Iteration convergence ─────────────────────────────
            if _history:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
            color:#c8a96e;font-weight:600;margin-bottom:0.8rem;">
  Iteration Convergence
</div>""", unsafe_allow_html=True)

                # ── Visual score progression bar chart ────────────────────────
                _scores = [h["ltn_score"] for h in _history]
                _bar_html = '<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:1rem;">' 
                for _hi, h in enumerate(_history):
                    _lsc  = h["ltn_score"]
                    _np   = sum(1 for r in h["audit"] if r.get("satisfies"))
                    _nt   = len(h["audit"]) or 1
                    _pct  = int(_lsc * 100)
                    _bc   = "#6dcea8" if _lsc >= _thresh else ("#e8c06d" if _lsc >= 0.5 else "#e8736d")
                    _delta = ""
                    if _hi > 0:
                        _d = _lsc - _scores[_hi-1]
                        _dsign = "+" if _d >= 0 else ""
                        _dc = "#6dcea8" if _d > 0 else ("#e8736d" if _d < 0 else "#888")
                        _delta = f'<span style="font-size:0.68rem;color:{_dc};margin-left:0.5rem;">{_dsign}{_d:.4f}</span>'
                    _is_last = h["attempt"] == len(_history)
                    _suffix = " ✓ FINAL" if (_is_last and _passed) else (" ✗ FINAL" if _is_last else " → rewrite")
                    _bar_html += f'''
<div style="display:flex;align-items:center;gap:8px;">
  <span style="font-family:DM Mono,monospace;font-size:0.72rem;color:rgba(232,228,220,0.5);
               min-width:52px;">Iter {h["attempt"]}</span>
  <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:16px;position:relative;">
    <div style="width:{_pct}%;background:{_bc};height:100%;border-radius:4px;
                transition:width 0.3s ease;"></div>
    <span style="position:absolute;left:6px;top:50%;transform:translateY(-50%);
                 font-size:0.65rem;color:#0c0e14;font-weight:700;line-height:1;">
      {_lsc:.4f}  {_np}/{_nt} rules
    </span>
  </div>
  {_delta}
  <span style="font-size:0.65rem;color:rgba(232,228,220,0.3);white-space:nowrap;">{_suffix}</span>
</div>'''
                _bar_html += '</div>'
                st.markdown(_bar_html, unsafe_allow_html=True)

                # Threshold reference
                st.caption(f"Pass threshold: {_thresh:.2f} | Bar width = LTN score proportion")

                # ── Per-iteration detail expanders ────────────────────────────
                for _hi, h in enumerate(_history):
                    _lsc  = h["ltn_score"]
                    _np   = sum(1 for r in h["audit"] if r.get("satisfies"))
                    _nt   = len(h["audit"])
                    _viols = h.get("violations", [])
                    _col_s = "#6dcea8" if _lsc >= _thresh else ("#e8c06d" if _lsc >= 0.5 else "#e8736d")
                    _is_last = h["attempt"] == len(_history)
                    _label = ("✅ Final — passed" if (_is_last and _passed)
                              else ("❌ Final — did not pass" if _is_last
                                    else f"⚠️ Triggered rewrite"))

                    with st.expander(f"Iteration {h['attempt']}: {_lsc:.4f}  ·  {_np}/{_nt} rules  ·  {_label}"):

                        # Rules that changed state vs previous iteration
                        if _hi > 0:
                            _prev_audit = {r.get("rule_id"): r for r in _history[_hi-1]["audit"]}
                            _curr_audit = {r.get("rule_id"): r for r in h["audit"]}
                            _improved, _degraded, _unchanged_fail = [], [], []
                            for _rid, _cr in _curr_audit.items():
                                _pr = _prev_audit.get(_rid)
                                if _pr:
                                    _pd, _cd = _pr.get("compliance_score",0), _cr.get("compliance_score",0)
                                    if _cd > _pd + 0.01:
                                        _improved.append((_cr.get("rule_display",""), _pd, _cd))
                                    elif _cd < _pd - 0.01:
                                        _degraded.append((_cr.get("rule_display",""), _pd, _cd))
                                    elif not _cr.get("satisfies"):
                                        _unchanged_fail.append(_cr.get("rule_display",""))

                            if _improved:
                                st.markdown("**Rules that improved this iteration:**")
                                for _rd, _pd, _cd in _improved:
                                    st.markdown(
                                        f'<div style="padding:0.3rem 0.7rem;background:rgba(109,206,168,0.07);'
                                        f'border-left:3px solid #6dcea8;border-radius:0 6px 6px 0;'
                                        f'font-size:0.79rem;margin-bottom:0.3rem;">'
                                        f'✅ {_rd.replace("<","&lt;")} &nbsp;'
                                        f'<span style="color:rgba(232,228,220,0.4);">'
                                        f'{_pd:.2f} → <strong style="color:#6dcea8">{_cd:.2f}</strong></span></div>',
                                        unsafe_allow_html=True)
                            if _degraded:
                                st.markdown("**Rules that got worse:**")
                                for _rd, _pd, _cd in _degraded:
                                    st.markdown(
                                        f'<div style="padding:0.3rem 0.7rem;background:rgba(232,115,109,0.07);'
                                        f'border-left:3px solid #e8736d;border-radius:0 6px 6px 0;'
                                        f'font-size:0.79rem;margin-bottom:0.3rem;">'
                                        f'📉 {_rd.replace("<","&lt;")} &nbsp;'
                                        f'<span style="color:rgba(232,228,220,0.4);">'
                                        f'{_pd:.2f} → <strong style="color:#e8736d">{_cd:.2f}</strong></span></div>',
                                        unsafe_allow_html=True)

                        # Violations that triggered the next rewrite
                        if _viols and not _is_last:
                            st.markdown(f"**{len(_viols)} rule(s) failed — system rewrote the draft:**")
                            for v in _viols:
                                vd = v.get("rule_display","").replace("<","&lt;")
                                ve = v.get("explanation","").replace("<","&lt;")
                                vs = v.get("compliance_score", 0)
                                st.markdown(f"""
<div style="padding:0.55rem 0.8rem;background:rgba(232,115,109,0.07);
     border-left:3px solid #e8736d;border-radius:0 8px 8px 0;
     margin-bottom:0.4rem;font-size:0.8rem;">
  <span style="color:#e8736d;font-weight:600;">{vd}</span>
  <span style="color:rgba(232,228,220,0.35);font-size:0.72rem;margin-left:0.4rem;">
    score: {vs:.2f}
  </span>
  <div style="color:rgba(232,228,220,0.5);margin-top:0.2rem;">{ve}</div>
</div>""", unsafe_allow_html=True)
                        elif _is_last and not _viols:
                            st.markdown('<p style="color:#6dcea8;font-size:0.83rem;">✅ All rules satisfied.</p>', unsafe_allow_html=True)
                        elif _is_last and _viols:
                            st.markdown('<p style="color:#e8736d;font-size:0.83rem;">❌ Some rules still failing at max iterations.</p>', unsafe_allow_html=True)

                        # Draft word count for this iteration
                        _wc = len(h.get("draft","").split())
                        if _hi > 0:
                            _pwc = len(_history[_hi-1].get("draft","").split())
                            _wdiff = _wc - _pwc
                            _wsign = "+" if _wdiff >= 0 else ""
                            st.caption(f"Draft: {_wc:,} words ({_wsign}{_wdiff} vs previous iteration)")
                        else:
                            st.caption(f"Draft: {_wc:,} words")

            # ── Section 5: Research Sources ───────────────────────────────────
            if _sources:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
            color:#c8a96e;font-weight:600;margin-bottom:0.8rem;">
  Where did the extra rules come from?
</div>""", unsafe_allow_html=True)
                research_rules = [r for r in _rules if r.get("source_name") not in ("User","")]
                st.markdown(f"""
<p style="font-size:0.83rem;color:rgba(232,228,220,0.55);line-height:1.7;margin-bottom:0.8rem;">
  In addition to your own rules, the system researched the topic online and derived
  <strong style="color:#c8a96e">{len(research_rules)} additional constraint(s)</strong>
  from the following source(s):
</p>""", unsafe_allow_html=True)
                for src in _sources:
                    sn  = src.get("source_name","Source")
                    ttl = src.get("title","")
                    ref = src.get("reference","")
                    ctx = src.get("context","")[:200]
                    ref_html = f'<a href="{ref}" target="_blank" style="color:#c8a96e;font-size:0.75rem;">{ref[:60]}…</a>' if ref and ref != "None" else ""
                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
     border-radius:10px;padding:0.85rem 1rem;margin-bottom:0.5rem;">
  <div style="font-size:0.82rem;font-weight:600;color:#e8e4dc;margin-bottom:0.3rem;">
    {sn} — {ttl}
  </div>
  <div style="font-size:0.77rem;color:rgba(232,228,220,0.45);line-height:1.6;margin-bottom:0.3rem;">
    "{ctx}…"
  </div>
  {ref_html}
</div>""", unsafe_allow_html=True)

            # ── Section 6: Download report as text ────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            _report_lines = [
                "NEUROSYMBOLIC VERIFIER -- TRANSPARENCY REPORT",
                "=" * 55,
                f"Run ID        : {_run_id}",
                f"LLM           : {res.get('llm_provider_name','Claude')} / {res.get('llm_model','')}",
                f"Verdict       : {_verdict}",
                f"LTN Score     : {_score:.4f}  (threshold: {_thresh:.2f})",
                f"Rules checked : {len(_audit)}  ({_n_pass} passed, {_n_fail} failed)",
                f"Iterations    : {_iters} / {_max_it}",
                "",
                "RULES",
                "-" * 40,
            ]
            for r in _audit:
                compliance = r.get("compliance_score", 1.0 if r.get("satisfies") else 0.0)
                status = "PASS" if r.get("satisfies") else ("PARTIAL" if compliance >= 0.5 else "FAIL")
                _report_lines += [
                    f"[{status} {compliance:.2f}] {r.get('rule_display','')}",
                    f"  Found    : {str(r.get('extracted_value_raw',''))[:100]}",
                    f"  Method   : {'Symbolic' if r.get('symbolic_check_used') else 'Semantic'}",
                    f"  Why      : {r.get('explanation','')}",
                    "",
                ]
            if _history and len(_history) > 1:
                _report_lines += ["ITERATION HISTORY", "-" * 40]
                for h in _history:
                    n_p = sum(1 for r in h["audit"] if r.get("satisfies"))
                    _report_lines.append(
                        f"Iteration {h['attempt']}: LTN={h['ltn_score']:.4f}  {n_p}/{len(h['audit'])} rules passed"
                    )
                _report_lines.append("")

            _report_txt = "\n".join(_report_lines)

            _rpt_col1, _rpt_col2 = st.columns(2)

            with _rpt_col1:
                st.download_button(
                    "⬇ Download Report (TXT)",
                    data      = _report_txt,
                    file_name = f"{_topic_slug}_report_{_run_id}.txt",
                    mime      = "text/plain",
                    key       = "dl_report_txt",
                    use_container_width = True,
                )

            with _rpt_col2:
                try:
                    from fpdf import FPDF
                    import re as _rre

                    class _RptPDF(FPDF):
                        def header(self):
                            self.set_font("Helvetica", "B", 8)
                            self.set_text_color(120, 120, 120)
                            _hcw = self.w - self.l_margin - self.r_margin
                            _htxt = (f"NeuroSymbolic Verifier -- Transparency Report  |  "
                                     f"{res.get('llm_provider_name','Claude')} / "
                                     f"{res.get('llm_model','')}  |  run {_run_id}")
                            self.set_x(self.l_margin)
                            self.multi_cell(_hcw, 5, _htxt, align="L")
                            self.ln(1)
                            self.set_draw_color(200, 169, 110)
                            self.set_line_width(0.4)
                            self.line(self.l_margin, self.get_y(),
                                      self.w - self.r_margin, self.get_y())
                            self.ln(3)
                        def footer(self):
                            self.set_y(-14)
                            self.set_font("Helvetica", "", 8)
                            self.set_text_color(160, 160, 160)
                            self.cell(0, 8, f"Page {self.page_no()}", align="C")

                    def _rsan(t):
                        _MAP = [
                            (chr(0x2014),"--"),(chr(0x2013),"-"),
                            (chr(0x2018),"'"),(chr(0x2019),"'"),
                            (chr(0x201C),'"'),(chr(0x201D),'"'),
                            (chr(0x2022),"-"),(chr(0x2264),"<="),(chr(0x2265),">="),
                            (chr(0x00B0)," deg"),(chr(0x00D7),"x"),(chr(0x2026),"..."),
                        ]
                        for uc, asc in _MAP:
                            t = t.replace(uc, asc)
                        return t.encode("latin-1", errors="replace").decode("latin-1")

                    _rpdf = _RptPDF()
                    _rpdf.set_margins(left=18, top=20, right=18)
                    _rpdf.set_auto_page_break(auto=True, margin=16)
                    _rpdf.add_page()
                    _cw = _rpdf.w - _rpdf.l_margin - _rpdf.r_margin

                    # Title
                    _rpdf.set_font("Helvetica","B",18)
                    _rpdf.set_text_color(27,58,92)
                    _rpdf.multi_cell(_cw, 9, "Transparency Report", align="L")
                    _rpdf.set_draw_color(46,117,182)
                    _rpdf.set_line_width(0.5)
                    _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                               _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                    _rpdf.ln(4)

                    # Summary box
                    _rpdf.set_font("Helvetica","B",10)
                    _rpdf.set_text_color(30,30,30)
                    _rpdf.set_fill_color(230,240,250)
                    _rpdf.multi_cell(_cw, 6,
                        _rsan(f"Verdict: {_verdict}   |   LTN Score: {_score:.4f} (threshold {_thresh:.2f})"
                              f"   |   {_n_pass}/{len(_audit)} rules passed   |   {_iters} iteration(s)"),
                        align="L", fill=True)
                    _rpdf.ln(4)

                    # Rules section
                    _rpdf.set_font("Helvetica","B",11)
                    _rpdf.set_text_color(46,117,182)
                    _rpdf.cell(_cw, 6, "Rule-by-Rule Results", ln=True)
                    _rpdf.set_line_width(0.3)
                    _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                               _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                    _rpdf.ln(3)

                    for _r in _audit:
                        _comp  = _r.get("compliance_score", 1.0 if _r.get("satisfies") else 0.0)
                        _stat  = "PASS" if _r.get("satisfies") else ("PARTIAL" if _comp>=0.5 else "FAIL")
                        _col   = (55,180,130) if _r.get("satisfies") else ((220,180,80) if _comp>=0.5 else (220,100,95))
                        _meth  = "Symbolic" if _r.get("symbolic_check_used") else "Semantic"
                        _disp  = _rsan(_r.get("rule_display",""))
                        _expl  = _rsan(_r.get("explanation",""))
                        _extr  = _rsan(str(_r.get("extracted_value_raw","N/A"))[:100])

                        # Status badge line
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.set_font("Helvetica","B",9)
                        _rpdf.set_text_color(*_col)
                        _rpdf.cell(28, 5, f"[{_stat} {_comp:.2f}]")
                        _rpdf.set_font("Helvetica","",9)
                        _rpdf.set_text_color(30,30,30)
                        _rpdf.multi_cell(_cw-28, 5, _disp, align="L")

                        # Details
                        _rpdf.set_font("Helvetica","",8)
                        _rpdf.set_text_color(90,90,90)
                        _rpdf.set_x(_rpdf.l_margin + 6)
                        _rpdf.multi_cell(_cw-6, 4, _rsan(f"Found: {_extr}"), align="L")
                        _rpdf.set_x(_rpdf.l_margin + 6)
                        _rpdf.multi_cell(_cw-6, 4,
                            _rsan(f"Method: {_meth} | {_expl}"), align="L")

                        # Compliance bar
                        # Safety: if near page bottom, let auto page break handle it
                        if _rpdf.get_y() > _rpdf.h - 30:
                            _rpdf.add_page()
                        _rpdf.set_x(_rpdf.l_margin + 6)
                        _bw = (_cw - 6) * 0.4
                        _rpdf.set_draw_color(220,220,220)
                        _rpdf.set_line_width(0.1)
                        _y_bar = _rpdf.get_y() + 1
                        _rpdf.rect(_rpdf.l_margin+6, _y_bar, _bw, 2.5)
                        _rpdf.set_fill_color(*_col)
                        _rpdf.rect(_rpdf.l_margin+6, _y_bar, _bw*_comp, 2.5, style="F")
                        _rpdf.ln(5)

                    # Iteration history — visual score bars + per-iter comparison
                    if _history and len(_history) >= 1:
                        _rpdf.ln(3)
                        if _rpdf.get_y() > _rpdf.h - 60: _rpdf.add_page()
                        _rpdf.set_font("Helvetica","B",11)
                        _rpdf.set_text_color(46,117,182)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 6,
                            f"Iteration Convergence ({len(_history)} iteration(s))", ln=True)
                        _rpdf.set_draw_color(46,117,182); _rpdf.set_line_width(0.3)
                        _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                                   _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                        _rpdf.ln(4)

                        # Draw horizontal bar chart: one bar per iteration
                        _bar_max_w = _cw * 0.55
                        _bar_h     = 7
                        _bar_gap   = 3
                        _lbl_w     = _cw * 0.18
                        _pct_w     = _cw * 0.12

                        for _h in _history:
                            _np  = sum(1 for _r in _h["audit"] if _r.get("satisfies"))
                            _nt  = len(_h["audit"]) or 1
                            _ls  = _h["ltn_score"]
                            _lc  = (55,180,130) if _ls>=_thresh else ((220,180,80) if _ls>=0.5 else (220,100,95))
                            _is_last = _h["attempt"] == len(_history)

                            if _rpdf.get_y() > _rpdf.h - 28: _rpdf.add_page()

                            _iy = _rpdf.get_y()
                            # Label
                            _rpdf.set_font("Helvetica","B" if _is_last else "",8)
                            _rpdf.set_text_color(*_lc)
                            _rpdf.set_xy(_rpdf.l_margin, _iy + 1)
                            _rpdf.cell(_lbl_w, _bar_h, f"Iter {_h['attempt']}", ln=False)
                            # Bar background
                            _bx = _rpdf.l_margin + _lbl_w
                            _rpdf.set_fill_color(230,230,230)
                            _rpdf.rect(_bx, _iy+1, _bar_max_w, _bar_h, style="F")
                            # Bar fill
                            _rpdf.set_fill_color(*_lc)
                            _rpdf.rect(_bx, _iy+1, _bar_max_w * min(_ls, 1.0), _bar_h, style="F")
                            # Score label
                            _rpdf.set_text_color(30,30,30)
                            _rpdf.set_font("Helvetica","",8)
                            _rpdf.set_xy(_bx + _bar_max_w + 3, _iy + 1)
                            _rpdf.cell(_pct_w, _bar_h, f"{_ls:.4f}  {_np}/{_nt}", ln=False)
                            _rpdf.ln(_bar_h + _bar_gap)

                            # Violations that triggered the next rewrite
                            _viols = _h.get("violations",[])
                            if _viols and not _is_last:
                                for _v in _viols[:3]:  # cap at 3 to save space
                                    if _rpdf.get_y() > _rpdf.h - 18: _rpdf.add_page()
                                    _rpdf.set_font("Helvetica","",7)
                                    _rpdf.set_text_color(200,80,80)
                                    _rpdf.set_x(_rpdf.l_margin + _lbl_w + 2)
                                    _rpdf.multi_cell(_cw - _lbl_w - 2, 4,
                                        _rsan(f"  Rewrite trigger: {_v.get('rule_display','')} (score {_v.get('compliance_score',0):.2f})"),
                                        align="L")
                            elif _is_last:
                                _rpdf.set_font("Helvetica","I",7)
                                _rpdf.set_text_color(55,180,130) if _passed else _rpdf.set_text_color(220,100,95)
                                _rpdf.set_x(_rpdf.l_margin + _lbl_w + 2)
                                _rpdf.cell(_cw - _lbl_w, 4,
                                    "Final -- PASSED" if _passed else "Final -- DID NOT PASS",
                                    ln=True)
                            _rpdf.ln(1)

                        # Threshold reference line note
                        _rpdf.set_font("Helvetica","I",7)
                        _rpdf.set_text_color(120,120,120)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 4,
                            f"Pass threshold: {_thresh:.2f} | Bars show LTN score (0.0 -- 1.0)",
                            ln=True)
                        _rpdf.ln(2)

                    # ── Column chart: LTN score per iteration ─────────────────
                    if _history and len(_history) >= 1:
                        if _rpdf.get_y() > _rpdf.h - 80: _rpdf.add_page()
                        _rpdf.ln(3)
                        _rpdf.set_font("Helvetica","B",11)
                        _rpdf.set_text_color(46,117,182)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 6, "LTN Score per Iteration (Column Chart)", ln=True)
                        _rpdf.set_draw_color(46,117,182); _rpdf.set_line_width(0.3)
                        _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                                   _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                        _rpdf.ln(4)

                        # Chart dimensions
                        _ch_h    = 55        # chart area height in mm
                        _ch_w    = _cw - 20  # chart area width
                        _ch_x    = _rpdf.l_margin + 14  # left offset for y-axis labels
                        _ch_y    = _rpdf.get_y()
                        _n_iters = len(_history)
                        _col_w   = min(20, _ch_w / max(_n_iters, 1))
                        _gap     = max(2, (_ch_w - _col_w * _n_iters) / max(_n_iters + 1, 1))

                        # Draw y-axis gridlines and labels (0.0, 0.25, 0.5, 0.75, 1.0)
                        for _yi, _yv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
                            _gy = _ch_y + _ch_h - _yv * _ch_h
                            _rpdf.set_draw_color(210,210,210); _rpdf.set_line_width(0.1)
                            _rpdf.line(_ch_x, _gy, _ch_x + _ch_w, _gy)
                            # Threshold line in gold
                            if abs(_yv - _thresh) < 0.13:
                                _rpdf.set_draw_color(200,169,110); _rpdf.set_line_width(0.4)
                                _rpdf.line(_ch_x, _ch_y + _ch_h - _thresh * _ch_h,
                                           _ch_x + _ch_w, _ch_y + _ch_h - _thresh * _ch_h)
                                _rpdf.set_font("Helvetica","I",6)
                                _rpdf.set_text_color(170,130,60)
                                _rpdf.set_xy(_ch_x + _ch_w + 1, _ch_y + _ch_h - _thresh*_ch_h - 2)
                                _rpdf.cell(12, 4, f"thr {_thresh:.2f}")
                            _rpdf.set_font("Helvetica","",6)
                            _rpdf.set_text_color(130,130,130)
                            _rpdf.set_xy(_rpdf.l_margin, _gy - 2)
                            _rpdf.cell(12, 4, f"{_yv:.2f}", align="R")

                        # Draw axes
                        _rpdf.set_draw_color(80,80,80); _rpdf.set_line_width(0.3)
                        _rpdf.line(_ch_x, _ch_y, _ch_x, _ch_y + _ch_h)            # y-axis
                        _rpdf.line(_ch_x, _ch_y + _ch_h, _ch_x + _ch_w, _ch_y + _ch_h)  # x-axis

                        # Draw columns
                        for _ci, _h in enumerate(_history):
                            _ls  = _h["ltn_score"]
                            _lc  = (55,180,130) if _ls>=_thresh else ((220,180,80) if _ls>=0.5 else (220,100,95))
                            _cx  = _ch_x + _gap + _ci * (_col_w + _gap)
                            _bar_top = _ch_y + _ch_h - _ls * _ch_h
                            _rpdf.set_fill_color(*_lc)
                            _rpdf.set_draw_color(200,200,200); _rpdf.set_line_width(0.1)
                            _rpdf.rect(_cx, _bar_top, _col_w, _ls * _ch_h,
                                       style="FD" if _ls > 0 else "D")
                            # Score label above bar
                            _rpdf.set_font("Helvetica","B",6)
                            _rpdf.set_text_color(*_lc)
                            _rpdf.set_xy(_cx - 1, _bar_top - 5)
                            _rpdf.cell(_col_w + 2, 4, f"{_ls:.3f}", align="C")
                            # Iteration label below x-axis
                            _rpdf.set_font("Helvetica","",7)
                            _rpdf.set_text_color(60,60,60)
                            _rpdf.set_xy(_cx - 1, _ch_y + _ch_h + 1)
                            _rpdf.cell(_col_w + 2, 4, f"I{_h['attempt']}", align="C")

                        _rpdf.set_xy(_rpdf.l_margin, _ch_y + _ch_h + 8)
                        _rpdf.set_font("Helvetica","I",7)
                        _rpdf.set_text_color(120,120,120)
                        _rpdf.cell(_cw, 4, "Y-axis: LTN score (0-1)  |  Gold line: pass threshold  |  Green = pass, Amber = marginal, Red = fail")
                        _rpdf.ln(6)

                    # ── Chart 3: Rule Compliance Grid ──────────────────────────
                    # Horizontal mini-bar for each rule — instant visual scan
                    # of which rules passed / marginal / failed.
                    if _audit:
                        if _rpdf.get_y() > _rpdf.h - 60: _rpdf.add_page()
                        _rpdf.ln(3)
                        _rpdf.set_font("Helvetica","B",11)
                        _rpdf.set_text_color(46,117,182)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 6,
                            f"Rule Compliance Grid ({len(_audit)} rules)", ln=True)
                        _rpdf.set_draw_color(46,117,182); _rpdf.set_line_width(0.3)
                        _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                                   _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                        _rpdf.ln(3)

                        _rg_bar_w   = _cw * 0.38   # width of the bar area
                        _rg_lbl_w   = _cw * 0.52   # width of rule label
                        _rg_pct_w   = _cw * 0.08   # width of score text
                        _rg_row_h   = 5.5

                        for _ri, _r in enumerate(_audit):
                            if _rpdf.get_y() > _rpdf.h - 14: _rpdf.add_page()
                            _comp  = _r.get("compliance_score",
                                            1.0 if _r.get("satisfies") else 0.0)
                            _col   = ((55,180,130) if _comp >= 0.8
                                      else ((220,180,80) if _comp >= 0.5
                                            else (220,100,95)))
                            _disp  = _rsan(_r.get("rule_display",""))[:52]
                            _meth  = "[sym]" if _r.get("symbolic_check_used") else "[sem]"
                            _stat  = "PASS" if _r.get("satisfies") else ("PART" if _comp>=0.5 else "FAIL")

                            _iy = _rpdf.get_y()

                            # Rule number + method tag
                            _rpdf.set_xy(_rpdf.l_margin, _iy)
                            _rpdf.set_font("Helvetica","B",6.5)
                            _rpdf.set_text_color(*_col)
                            _rpdf.cell(10, _rg_row_h, f"R{_ri+1}", ln=False)

                            # Rule label
                            _rpdf.set_font("Helvetica","",7)
                            _rpdf.set_text_color(50,50,50)
                            _rpdf.set_xy(_rpdf.l_margin + 10, _iy)
                            _rpdf.cell(_rg_lbl_w - 10, _rg_row_h, _disp, ln=False)

                            # Bar background
                            _bx = _rpdf.l_margin + _rg_lbl_w
                            _rpdf.set_fill_color(225,225,225)
                            _rpdf.rect(_bx, _iy+1, _rg_bar_w, _rg_row_h-2, style="F")

                            # Bar fill
                            _rpdf.set_fill_color(*_col)
                            _rpdf.rect(_bx, _iy+1, _rg_bar_w*min(_comp,1.0),
                                       _rg_row_h-2, style="F")

                            # Score + method
                            _rpdf.set_font("Helvetica","B",6.5)
                            _rpdf.set_text_color(30,30,30)
                            _rpdf.set_xy(_bx + _rg_bar_w + 2, _iy)
                            _rpdf.cell(_rg_pct_w, _rg_row_h,
                                f"{_comp:.2f} {_meth}", ln=True)

                        _rpdf.ln(2)
                        _rpdf.set_font("Helvetica","I",7)
                        _rpdf.set_text_color(120,120,120)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 4,
                            "Bar width = compliance score (0-1)  |  [sym] = symbolic math check, [sem] = AI semantic check  |"
                            "  Green >= 0.80, Amber >= 0.50, Red < 0.50")
                        _rpdf.ln(6)
                    if _history and len(_history) >= 1:
                        if _rpdf.get_y() > _rpdf.h - 50: _rpdf.add_page()
                        _rpdf.ln(2)
                        _rpdf.set_font("Helvetica","B",11)
                        _rpdf.set_text_color(46,117,182)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 6, "Per-Iteration Failure Analysis", ln=True)
                        _rpdf.set_draw_color(46,117,182); _rpdf.set_line_width(0.3)
                        _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                                   _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                        _rpdf.ln(3)

                        for _h in _history:
                            if _rpdf.get_y() > _rpdf.h - 35: _rpdf.add_page()
                            _np   = sum(1 for _r in _h["audit"] if _r.get("satisfies"))
                            _nt   = len(_h["audit"]) or 1
                            _ls   = _h["ltn_score"]
                            _lc   = (55,180,130) if _ls>=_thresh else ((220,180,80) if _ls>=0.5 else (220,100,95))
                            _viols = _h.get("violations",[])
                            _is_last = _h["attempt"] == len(_history)

                            # ── Iteration header ─────────────────────────────
                            _rpdf.set_x(_rpdf.l_margin)
                            _rpdf.set_font("Helvetica","B",9)
                            _rpdf.set_text_color(*_lc)
                            _outcome = ("PASSED" if (_is_last and _passed)
                                        else ("FAILED (max iterations)" if _is_last else "TRIGGERED REWRITE"))
                            _rpdf.cell(22, 5, f"Iter {_h['attempt']}")
                            _rpdf.set_font("Helvetica","",9)
                            _rpdf.set_text_color(30,30,30)
                            # Word count
                            _wc = len(_h.get("draft","").split())
                            _rpdf.multi_cell(_cw-22, 5,
                                _rsan(f"LTN {_ls:.4f} | {_np}/{_nt} rules | {_wc:,} words | {_outcome}"),
                                align="L")

                            # ── Rule state changes (improved / degraded) ──────
                            _hi_idx = _history.index(_h)
                            if _hi_idx > 0:
                                _prev_audit = {r.get("rule_id"): r
                                               for r in _history[_hi_idx-1]["audit"]}
                                _curr_audit = {r.get("rule_id"): r
                                               for r in _h["audit"]}
                                _improved_r, _degraded_r = [], []
                                for _rid, _cr in _curr_audit.items():
                                    _pr = _prev_audit.get(_rid)
                                    if _pr:
                                        _pd = _pr.get("compliance_score", 0)
                                        _cd = _cr.get("compliance_score", 0)
                                        _rd = _cr.get("rule_display","")[:50]
                                        if _cd > _pd + 0.01:
                                            _improved_r.append(
                                                f"  + {_rd}: {_pd:.2f}->{_cd:.2f}")
                                        elif _cd < _pd - 0.01:
                                            _degraded_r.append(
                                                f"  - {_rd}: {_pd:.2f}->{_cd:.2f}")
                                if _improved_r:
                                    _rpdf.set_x(_rpdf.l_margin + 6)
                                    _rpdf.set_font("Helvetica","B",7)
                                    _rpdf.set_text_color(55,180,130)
                                    _rpdf.cell(_cw-6, 4, "Rules improved:", ln=True)
                                    for _rline in _improved_r[:4]:
                                        _rpdf.set_x(_rpdf.l_margin + 10)
                                        _rpdf.set_font("Helvetica","",7)
                                        _rpdf.set_text_color(55,160,110)
                                        _rpdf.multi_cell(_cw-10, 4, _rsan(_rline), align="L")
                                if _degraded_r:
                                    _rpdf.set_x(_rpdf.l_margin + 6)
                                    _rpdf.set_font("Helvetica","B",7)
                                    _rpdf.set_text_color(220,100,95)
                                    _rpdf.cell(_cw-6, 4, "Rules degraded:", ln=True)
                                    for _rline in _degraded_r[:4]:
                                        _rpdf.set_x(_rpdf.l_margin + 10)
                                        _rpdf.set_font("Helvetica","",7)
                                        _rpdf.set_text_color(200,80,80)
                                        _rpdf.multi_cell(_cw-10, 4, _rsan(_rline), align="L")

                            # ── Violations that triggered next rewrite ────────
                            if not _viols:
                                _rpdf.set_x(_rpdf.l_margin + 6)
                                _rpdf.set_font("Helvetica","I",8)
                                _rpdf.set_text_color(55,180,130)
                                _rpdf.cell(_cw-6, 4, "All rules satisfied.", ln=True)
                            else:
                                for _v in _viols:
                                    if _rpdf.get_y() > _rpdf.h - 22: _rpdf.add_page()
                                    _vd   = _rsan(_v.get("rule_display",""))
                                    _ve   = _rsan(_v.get("explanation",""))
                                    _vs   = _v.get("compliance_score",0)
                                    _vmeth = "Symbolic" if _v.get("symbolic_check_used") else "Semantic"
                                    _vextr = _rsan(str(_v.get("extracted_value_raw",""))[:80])

                                    _rpdf.set_x(_rpdf.l_margin + 6)
                                    _rpdf.set_font("Helvetica","B",8)
                                    _rpdf.set_text_color(200,80,80)
                                    _rpdf.multi_cell(_cw-6, 4,
                                        _rsan(f"FAIL ({_vs:.2f}) -- {_vd}"), align="L")
                                    _rpdf.set_x(_rpdf.l_margin + 10)
                                    _rpdf.set_font("Helvetica","",7)
                                    _rpdf.set_text_color(80,80,80)
                                    _rpdf.multi_cell(_cw-10, 4,
                                        _rsan(f"Found: {_vextr}"), align="L")
                                    _rpdf.set_x(_rpdf.l_margin + 10)
                                    _rpdf.multi_cell(_cw-10, 4,
                                        _rsan(f"Why: {_ve}"), align="L")
                                    _rpdf.set_x(_rpdf.l_margin + 10)
                                    _rpdf.set_text_color(120,120,120)
                                    _rpdf.cell(_cw-10, 4,
                                        _rsan(f"Method: {_vmeth}"), ln=True)
                                    if not _is_last:
                                        _rpdf.set_x(_rpdf.l_margin + 10)
                                        _rpdf.set_font("Helvetica","I",7)
                                        _rpdf.set_text_color(150,100,50)
                                        _rpdf.cell(_cw-10, 4,
                                            "-> System rewrote draft based on this failure",
                                            ln=True)
                                    _rpdf.ln(1)

                            # ── Draft excerpt (first 300 chars) ───────────────
                            _excerpt = _h.get("draft","")[:300].replace("\n"," ").strip()
                            if _excerpt:
                                if _rpdf.get_y() > _rpdf.h - 20: _rpdf.add_page()
                                _rpdf.set_x(_rpdf.l_margin + 6)
                                _rpdf.set_font("Helvetica","I",7)
                                _rpdf.set_text_color(110,110,110)
                                _rpdf.set_fill_color(248,248,248)
                                _rpdf.multi_cell(_cw-6, 3.8,
                                    _rsan(f'Draft excerpt: "{_excerpt}..."'),
                                    align="L", fill=True)
                            _rpdf.ln(3)

                    # ── Reference URLs — all sources, full clickable URLs ──────
                    # Collect every URL from sources (Wikipedia, DDG, Web,
                    # Google, Custom URLs).  Print the FULL url using
                    # rpdf.write() which supports the link= parameter AND
                    # auto-wraps long URLs — no truncation ever.
                    _all_refs = []
                    for _src in _sources:
                        _ref = (_src.get("reference","") or
                                _src.get("source","") or "").strip()
                        if _ref and _ref not in ("None","none",""):
                            _all_refs.append({
                                "source_name": _src.get("source_name","Source"),
                                "title"      : _src.get("title",""),
                                "url"        : _ref,
                                "context"    : _src.get("context",""),
                            })

                    if _all_refs:
                        if _rpdf.get_y() > _rpdf.h - 50: _rpdf.add_page()
                        _rpdf.ln(2)
                        _rpdf.set_font("Helvetica","B",11)
                        _rpdf.set_text_color(46,117,182)
                        _rpdf.set_x(_rpdf.l_margin)
                        _rpdf.cell(_cw, 6,
                            f"Research Sources & Reference URLs ({len(_all_refs)})",
                            ln=True)
                        _rpdf.set_draw_color(46,117,182); _rpdf.set_line_width(0.3)
                        _rpdf.line(_rpdf.l_margin, _rpdf.get_y(),
                                   _rpdf.w-_rpdf.r_margin, _rpdf.get_y())
                        _rpdf.ln(3)

                        for _si, _src in enumerate(_all_refs):
                            if _rpdf.get_y() > _rpdf.h - 45: _rpdf.add_page()

                            # Source name + title header
                            _rpdf.set_x(_rpdf.l_margin)
                            _rpdf.set_font("Helvetica","B",9)
                            _rpdf.set_text_color(30,30,30)
                            _rpdf.multi_cell(_cw, 5,
                                _rsan(f"[{_si+1}] {_src['source_name']} -- {_src['title']}"),
                                align="L")

                            # Context snippet (200 chars)
                            _ctx_snip = _src["context"][:200].strip()
                            if _ctx_snip:
                                _rpdf.set_x(_rpdf.l_margin)
                                _rpdf.set_font("Helvetica","I",8)
                                _rpdf.set_text_color(100,100,100)
                                _rpdf.multi_cell(_cw, 4,
                                    _rsan(f'"{_ctx_snip}..."'), align="L")

                            # Full URL — clickable, blue, wraps naturally
                            # fpdf2 write() supports link= and auto-wraps
                            _url = _src["url"]
                            if _url:
                                _rpdf.set_x(_rpdf.l_margin)
                                _rpdf.set_font("Courier","",7.5)
                                _rpdf.set_text_color(27,100,200)
                                try:
                                    _rpdf.write(4.5, _url, link=_url)
                                    _rpdf.ln(4.5)
                                except Exception:
                                    # Fallback: multi_cell without link if write fails
                                    _rpdf.multi_cell(_cw, 4, _rsan(_url), align="L")
                            _rpdf.ln(3)

                    _rpt_pdf_bytes = bytes(_rpdf.output())
                    st.download_button(
                        "⬇ Download Report (PDF)",
                        data      = _rpt_pdf_bytes,
                        file_name = f"{_topic_slug}_report_{_run_id}.pdf",
                        mime      = "application/pdf",
                        key       = "dl_report_pdf",
                        use_container_width = True,
                    )
                except Exception as _rpt_pdf_err:
                    st.button("⬇ PDF unavailable", disabled=True,
                              key="dl_report_pdf_disabled",
                              help=str(_rpt_pdf_err),
                              use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Start New Run", key="reset_results",
                    use_container_width=True):
            for _k in ["results","brain_records","_pipeline_running"]:
                if _k in st.session_state:
                    del st.session_state[_k]
            st.session_state.results       = None
            st.session_state.brain_records = {}
            st.session_state["_pipeline_running"] = False
            # Reset Qdrant client — same fix as hard reset above
            try:
                import m3_vector_db as _m3_reset
                _m3_reset.reset_client()
            except Exception:
                pass
            st.rerun()
