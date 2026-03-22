from flask import Flask, request, jsonify, render_template_string
import psycopg2
import anthropic
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL  = os.getenv("DATABASE_URL")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
INDEX_FILE    = r"C:\email-intelligence\product_index.json"

client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
app      = Flask(__name__)
histories = {}

# Carica indice prodotti in memoria all'avvio
print("Caricamento indice prodotti...")
try:
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        PRODUCT_INDEX = json.load(f)
    print(f"Indice caricato: {len(PRODUCT_INDEX)} prodotti")
except Exception as e:
    print(f"ATTENZIONE: indice non trovato ({e})")
    PRODUCT_INDEX = []

SINONIMI = {
    "teglia":         ["teglie","stampo","stampi","placca","placche","formina"],
    "teglie":         ["teglia","stampo","stampi","placca","placche"],
    "baguette":       ["filone","filoncino","pane lungo","sfilatino"],
    "forato":         ["forata","forati","forate","perforato","perforata","microforato"],
    "alluminio":      ["alluminio","allum","alum","allumino"],
    "telaio":         ["telai","traversino","traversini","carrello","struttura"],
    "traversino":     ["telaio","telai","traversini","carrello"],
    "pizza":          ["pizze","pizzeria","focaccia","pinsa"],
    "pane":           ["panificio","panetteria","panetto","pagnotta","filone"],
    "forno":          ["forni","cottura","infornare","cuocere"],
    "arrotondatrice": ["arrotondatrici","ballmatic","arrotondare","formatura"],
    "ballmatic":      ["arrotondatrice","arrotondatrici","formatura palle"],
    "impastatrice":   ["impastatrici","impasto","impastatrice","spirale"],
    "sfogliatrice":   ["sfogliatrici","sfoglia","laminatoio","stendi"],
    "sottovuoto":     ["confezionamento","conservazione","termosaldatura"],
    "lievitazione":   ["lievitare","lievito","fermalievitazione","cella","fermabiga"],
    "fermalievitazione": ["cella","lievitazione","fermabiga","armadio lievitazione"],
    "stagionatura":   ["maturazione","stagionare","celle stagionatura"],
    "abbattitore":    ["abbattimento","raffreddamento rapido","shock termico"],
    "impastatrice":   ["impastatrici","spirale","forcella","bracci tuffanti"],
    "spezzatrice":    ["spezza","porzionatura","divisore"],
    "formatrice":     ["formare","formatura","cilindro"],
    "60x40":          ["60 x 40","600x400","60x40 cm","teglia standard"],
    "40x30":          ["40 x 30","400x300"],
    "piana":          ["piane","piatta","piatte","liscia","lisce"],
    "bordo":          ["bordato","bordati","bordata","bordate","con bordo"],
}

def normalizza(testo):
    if not testo:
        return ""
    t = testo.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def espandi_query(query):
    parole = normalizza(query).split()
    espanse = set(parole)
    for p in parole:
        if p in SINONIMI:
            espanse.update(SINONIMI[p])
    return list(espanse)

def cerca_prodotti_candidati(query, limit=12):
    if not PRODUCT_INDEX:
        return []
    parole_query = espandi_query(query)
    scores = []
    for prod in PRODUCT_INDEX:
        keywords  = set(prod.get("keywords", []))
        nome_norm = normalizza(prod.get("name", ""))
        cat_norm  = normalizza(prod.get("category", ""))
        score = 0
        for parola in parole_query:
            if parola in nome_norm: score += 10
            if parola in cat_norm:  score += 5
            if parola in keywords:  score += 2
        matches_nome = sum(1 for p in parole_query if p in nome_norm)
        if matches_nome >= 2:
            score += matches_nome * 5
        if score > 0:
            scores.append((score, prod))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scores[:limit]]


def filtra_con_claude(query, candidati, limit=4):
    if not candidati:
        return []
    if len(candidati) <= 2:
        return candidati[:limit]
    lista = ""
    for i, p in enumerate(candidati):
        lista += f"{i+1}. {p['name']}"
        if p.get("dims"):  lista += f" | {p['dims']}"
        if p.get("attrs"): lista += f" | {p['attrs'][:80]}"
        if p.get("desc"):  lista += f" | {p['desc'][:80]}"
        lista += "\n"
    prompt = (
        f"Un cliente ha chiesto: \"{query}\"\n\n"
        f"Prodotti nel catalogo Starpizza (attrezzature professionali):\n"
        f"{lista}\n"
        f"Seleziona SOLO i numeri dei prodotti davvero pertinenti. "
        f"Massimo {limit}. "
        f"Escludi prodotti per uso domestico se la richiesta e professionale. "
        f"Escludi prodotti che contengono la parola cercata ma non sono quello che cerca. "
        f"Rispondi SOLO con numeri separati da virgola, esempio: 1,3"
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        risposta = msg.content[0].text.strip()
        numeri = [int(x.strip()) - 1 for x in risposta.split(",") if x.strip().isdigit()]
        selezionati = [candidati[n] for n in numeri if 0 <= n < len(candidati)]
        return selezionati[:limit]
    except Exception as e:
        print(f"Errore filtro Claude: {e}")
        return candidati[:limit]


def cerca_prodotti(query, limit=4):
    candidati = cerca_prodotti_candidati(query, limit=12)
    if not candidati:
        return []
    return filtra_con_claude(query, candidati, limit=limit)

HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>Sophie - Starpizza</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; font-size: 16px; }
header { background: #1a1a2e; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2a2a4a; }
.hlogo { display: flex; align-items: center; gap: 10px; }
.hav { width: 38px; height: 38px; background: linear-gradient(135deg, #a78bfa, #7c3aed); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #fff; font-size: 1.1rem; }
.hname h1 { font-size: 1.15rem; margin: 0; }
.hname p { font-size: 0.85rem; color: #6dbf6d; margin: 0; }
.hright { display: flex; align-items: center; gap: 8px; }
.hright span { font-size: 0.82rem; color: #444; }
#rbtn { background: none; border: 1px solid #333; color: #666; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
#rbtn:hover { border-color: #a78bfa; color: #a78bfa; }
#chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.row { display: flex; align-items: flex-end; gap: 8px; }
.row.user { flex-direction: row-reverse; }
.mav { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; font-weight: bold; }
.mav.s { background: linear-gradient(135deg, #a78bfa, #7c3aed); color: #fff; }
.mav.u { background: #2d2d5e; color: #aaa; }
.bub { max-width: 70%; padding: 13px 17px; border-radius: 14px; font-size: 1.05rem; line-height: 1.75; white-space: pre-wrap; }
.bub.s { background: #1e1e1e; border: 1px solid #2a2a4a; border-bottom-left-radius: 3px; }
.bub.u { background: #2d2d5e; border-bottom-right-radius: 3px; }
.bub.loading { color: #555; font-style: italic; }
.src { margin-top: 7px; padding-top: 7px; border-top: 1px solid #222; font-size: 0.8rem; color: #555; display: flex; flex-wrap: wrap; gap: 4px; }
.tag { display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 0.78rem; }
.td { background: #1a2a1a; color: #4a9a4a; }
.te { background: #1a1a2e; color: #6a6abf; }
.tn { background: #2a1a1a; color: #9a4a4a; }
.prow { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.pcard { background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 9px 14px; font-size: 0.88rem; text-decoration: none; color: #a78bfa; display: inline-block; transition: all 0.2s; }
.pcard:hover { background: #1e2d5e; border-color: #a78bfa; }
.pcard strong { display: block; color: #e0e0e0; font-size: 0.92rem; margin-top: 2px; }
#foot { padding: 14px 20px; background: #111; border-top: 1px solid #1e1e1e; display: flex; gap: 10px; }
#inp { flex: 1; background: #1e1e1e; border: 1px solid #2a2a2a; color: #e0e0e0; padding: 12px 18px; border-radius: 22px; font-size: 1.05rem; outline: none; }
#inp:focus { border-color: #7c3aed; }
#sbtn { background: linear-gradient(135deg, #a78bfa, #7c3aed); color: #fff; border: none; width: 44px; height: 44px; border-radius: 50%; cursor: pointer; font-size: 1.2rem; flex-shrink: 0; }
#sbtn:disabled { opacity: 0.3; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="hlogo">
    <div class="hav">S</div>
    <div class="hname"><h1>Sophie</h1><p>Online &mdash; Assistente virtuale Starpizza</p></div>
  </div>
  <div class="hright">
    <span>powered by Arcanum</span>
    <button id="rbtn" onclick="resetChat()">Nuova chat</button>
  </div>
</header>
<div id="chat"></div>
<div id="foot">
  <input id="inp" type="text" placeholder="Scrivi un messaggio..." autocomplete="off">
  <button id="sbtn" onclick="send()">&#9658;</button>
</div>
<script>
var cid = Math.random().toString(36).slice(2);

function buildBubble(text, sources, emails, products) {
  var escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  var src = '<div class="src">';
  if (sources && sources.length) {
    for (var i = 0; i < sources.length; i++) src += '<span class="tag td">' + sources[i] + '</span>';
  }
  if (emails > 0) src += '<span class="tag te">' + emails + ' email</span>';
  // nessuna fonte rimossa — non mostrare nulla se non ci sono fonti
  src += '</div>';
  var phtml = '';
  if (products && products.length > 0) {
    phtml = '<div class="prow">';
    for (var j = 0; j < products.length; j++) {
      phtml += '<a class="pcard" href="' + products[j].url + '" target="_blank">Acquista<strong>' + products[j].name + '</strong></a>';
    }
    phtml += '</div>';
  }
  return escaped + src + phtml;
}

function addMsg(text, role, sources, emails, products) {
  var chat = document.getElementById('chat');
  var row = document.createElement('div');
  row.className = role === 'u' ? 'row user' : 'row';
  var av = document.createElement('div');
  av.className = 'mav ' + role;
  av.textContent = role === 's' ? 'S' : 'U';
  var bub = document.createElement('div');
  bub.className = 'bub ' + role;
  if (role === 's') {
    bub.innerHTML = buildBubble(text, sources, emails, products);
  } else {
    bub.textContent = text;
  }
  row.appendChild(av);
  row.appendChild(bub);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bub;
}

window.addEventListener('load', function() {
  fetch('/welcome').then(function(r){return r.json();}).then(function(d){
    var chat = document.getElementById('chat');
    var row = document.createElement('div');
    row.className = 'row';
    var av = document.createElement('div');
    av.className = 'mav s';
    av.textContent = 'S';
    var bub = document.createElement('div');
    bub.className = 'bub s';
    bub.innerHTML = d.text;
    row.appendChild(av);
    row.appendChild(bub);
    chat.appendChild(row);
  });
});

document.getElementById('inp').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') send();
});

function resetChat() {
  fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cid:cid})});
  document.getElementById('chat').innerHTML = '';
  fetch('/welcome').then(function(r){return r.json();}).then(function(d){
    var chat = document.getElementById('chat');
    var row = document.createElement('div');
    row.className = 'row';
    var av = document.createElement('div');
    av.className = 'mav s';
    av.textContent = 'S';
    var bub = document.createElement('div');
    bub.className = 'bub s';
    bub.innerHTML = d.text;
    row.appendChild(av);
    row.appendChild(bub);
    chat.appendChild(row);
  });
}

function send() {
  var inp = document.getElementById('inp');
  var sbtn = document.getElementById('sbtn');
  var text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  sbtn.disabled = true;
  addMsg(text, 'u', [], 0, []);
  var loading = addMsg('...', 's', [], 0, []);
  loading.classList.add('loading');
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, cid: cid})
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    loading.classList.remove('loading');
    loading.innerHTML = buildBubble(data.response, data.doc_sources, data.email_count, data.products);
    document.getElementById('chat').scrollTop = 99999;
  })
  .catch(function() {
    loading.classList.remove('loading');
    loading.textContent = 'Errore di connessione. Riprova.';
  })
  .finally(function() {
    sbtn.disabled = false;
    inp.focus();
  });
}
</script>
</body>
</html>"""


def get_docs(query):
    try:
        conn  = psycopg2.connect(DATABASE_URL)
        cur   = conn.cursor()
        words = [w.lower() for w in query.split() if len(w) > 2]
        docs  = []
        seen  = set()
        for word in words[:10]:
            cur.execute("""
                SELECT file_name, category_name, product_name, content
                FROM documents
                WHERE LOWER(content) LIKE %s OR LOWER(file_name) LIKE %s
                   OR LOWER(product_name) LIKE %s OR LOWER(category_name) LIKE %s
                LIMIT 3
            """, (f"%{word}%",)*4)
            for row in cur.fetchall():
                if row[0] not in seen:
                    docs.append(row); seen.add(row[0])
            if len(docs) >= 4: break
        cur.close(); conn.close()
        return docs[:4]
    except Exception as e:
        print(f"Errore docs: {e}"); return []


def get_emails(query, limit=6):
    try:
        conn  = psycopg2.connect(DATABASE_URL)
        cur   = conn.cursor()
        STOP  = {"come","cosa","sono","avete","anche","pero","quando","dove",
                 "questo","questa","questi","queste","molto","poco","bene",
                 "fare","avere","essere","dalla","nella","dello","della"}
        words = [w.lower() for w in query.split() if len(w) > 4 and w.lower() not in STOP]
        if not words:
            return []
        conditions = " OR ".join(
            ["(LOWER(subject) LIKE %s OR LOWER(body_clean) LIKE %s)"] * len(words)
        )
        params = []
        for w in words:
            params += [f"%{w}%", f"%{w}%"]
        cur.execute(
            "SELECT subject, body_clean, category, sentiment FROM emails "
            "WHERE folder='inbox' AND status='classified' "
            "AND (" + conditions + ") "
            "ORDER BY received_at DESC LIMIT %s",
            params + [limit]
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f"Errore emails: {e}"); return []


def salva_chat(cid, ruolo, testo):
    """Salva ogni messaggio della conversazione nel database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id          SERIAL PRIMARY KEY,
                session_id  VARCHAR(64),
                ruolo       VARCHAR(16),
                testo       TEXT,
                creato_il   TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute(
            "INSERT INTO chats (session_id, ruolo, testo) VALUES (%s, %s, %s)",
            (cid, ruolo, testo[:4000])
        )
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"Errore salva_chat: {e}")


@app.route("/welcome")
def welcome():
    return jsonify({"text": "Ciao! Sono Sophie, l'assistente virtuale di Starpizza.<br><br>Come posso aiutarti oggi?"})


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/reset", methods=["POST"])
def reset():
    data = request.json or {}
    histories[data.get("cid", "default")] = []
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json or {}
    message = data.get("message", "").strip()
    cid     = data.get("cid", "default")

    if not message:
        return jsonify({"response": "...", "doc_sources": [], "email_count": 0, "products": []})

    if cid not in histories:
        histories[cid] = []
    history = histories[cid]

    last = [m["content"] for m in history if m["role"] == "user"][-2:]
    q    = " ".join(last + [message])

    docs     = get_docs(q)
    emails   = get_emails(q)
    products = cerca_prodotti(q, limit=4)

    docs_ctx    = ""
    doc_sources = []
    if docs:
        docs_ctx = "\n\n=== DOCUMENTAZIONE TECNICA ===\n"
        for fname, cat, prod, content in docs:
            docs_ctx += f"\n[{fname} - {prod or cat}]\n{content[:2000]}\n"
            doc_sources.append(fname)

    email_ctx = ""
    if emails:
        email_ctx = f"\n\n=== STORICO EMAIL CLIENTI ({len(emails)}) ===\n"
        for subj, body, cat, sent in emails:
            email_ctx += f"\n[{subj} | {cat} | {sent}]\n{(body or '')[:300]}\n"

    products_ctx = ""
    prod_note    = ""
    if products:
        products_ctx = "\n\n=== PRODOTTI STARPIZZA TROVATI ===\n"
        for p in products:
            products_ctx += f"\n- {p['name']}"
            if p.get('dims'):  products_ctx += f" | Dimensioni: {p['dims']}"
            if p.get('attrs'): products_ctx += f" | {p['attrs']}"
            if p.get('desc'):  products_ctx += f"\n  {p['desc']}"
            products_ctx += f"\n  URL: {p['url']}\n"
        prod_note = (
            f"\nHai {len(products)} prodotti pertinenti trovati. "
            "Menzionali quando utile. NON indicare mai prezzi. "
            "NON menzionare mai il nome del produttore o brand costruttore."
        )

    if docs:
        note = "Hai documentazione tecnica - usala con precisione."
    elif emails:
        note = f"Hai {len(emails)} email storiche simili - analizzale e riassumi."
    else:
        note = "Non hai info specifiche nel database. Dillo con naturalezza e suggerisci di contattare Starpizza."

    # Rileva lingua del messaggio
    msg_lower = message.lower()
    lang_hint = ""
    if any(w in msg_lower for w in ["the ","is ","are ","have","what","where","how ","can ","i ","we ","my ","your "]):
        lang_hint = "CRITICAL RULE: The customer is writing in ENGLISH. You MUST reply in ENGLISH only. Never use Italian.\n\n"
    elif any(w in msg_lower for w in ["bonjour","merci","vous ","je ","est ","les ","des ","que ","pour "]):
        lang_hint = "REGLE CRITIQUE: Le client ecrit en francais. Tu DOIS repondre en FRANCAIS uniquement.\n\n"
    elif any(w in msg_lower for w in ["hola","gracias","tiene","como ","por favor","para ","esto "]):
        lang_hint = "REGLA CRITICA: El cliente escribe en espanol. DEBES responder en ESPANOL unicamente.\n\n"
    elif any(w in msg_lower for w in ["danke","bitte","haben","ich ","sie ","die ","der ","das "]):
        lang_hint = "KRITISCHE REGEL: Der Kunde schreibt auf Deutsch. Du MUSST auf DEUTSCH antworten.\n\n"

    system = (
        "Sei Sophie, l assistente virtuale di Starpizza - "
        "attrezzature professionali per pizzerie, panifici e ristorazione.\n\n"
        + lang_hint +
        "PERSONALITA:\n"
        "- Gentile, calda, competente come una consulente esperta\n"
        "- Naturale e conversazionale, senza elenchi puntati\n"
        "- Breve e diretta come in una chat reale\n"
        "- Fai domande di follow-up quando serve\n"
        "- Adatta il registro al cliente\n"
        "- REGOLA ASSOLUTA: rispondi SEMPRE nella stessa lingua del cliente\n"
        "- Ricorda il filo della conversazione\n\n"
        "REGOLE:\n"
        f"- {note}{prod_note}\n"
        "- NON inventare prezzi, dimensioni o specifiche tecniche\n"
        "- NON menzionare mai il nome del produttore o brand\n"
        "- Per prezzi rimanda al sito o al team Starpizza\n"
        + docs_ctx + email_ctx + products_ctx
    )

    history.append({"role": "user", "content": message})

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            temperature=0.4,
            system=system,
            messages=history
        )
        response = msg.content[0].text.strip()
        history.append({"role": "assistant", "content": response})
        if len(history) > 40:
            histories[cid] = history[-40:]
        # Salva conversazione nel DB
        salva_chat(cid, "cliente", message)
        salva_chat(cid, "sophie", response)
        return jsonify({
            "response":    response,
            "doc_sources": doc_sources,
            "email_count": len(emails),
            "products":    products
        })
    except Exception as e:
        history.pop()
        return jsonify({"response": f"Errore: {str(e)}", "doc_sources": [], "email_count": 0, "products": []})


@app.route("/admin/chat")
def admin_chat():
    """Pagina privata per monitorare le conversazioni di Sophie."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY, session_id VARCHAR(64),
                ruolo VARCHAR(16), testo TEXT, creato_il TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            SELECT session_id, ruolo, testo, creato_il
            FROM chats
            ORDER BY creato_il DESC
            LIMIT 200
        """)
        righe = cur.fetchall()
        cur.close(); conn.close()
    except Exception as e:
        return f"Errore DB: {e}"

    # Raggruppa per sessione
    sessioni = {}
    for sid, ruolo, testo, ts in righe:
        if sid not in sessioni:
            sessioni[sid] = []
        sessioni[sid].append((ruolo, testo, ts))

    html = """
    <html><head><meta charset="UTF-8"><title>Sophie - Monitor Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        h1 { color: #c0392b; }
        .sessione { background: white; border-radius: 10px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .sessione h3 { color: #888; font-size: 0.85rem; margin-bottom: 12px; }
        .msg { padding: 8px 12px; border-radius: 8px; margin: 6px 0; max-width: 80%; font-size: 0.95rem; }
        .cliente { background: #c0392b; color: white; margin-left: auto; text-align: right; }
        .sophie { background: #f0f0f0; color: #222; }
        .ts { font-size: 0.7rem; color: #aaa; margin-top: 2px; }
        .wrap { display: flex; flex-direction: column; }
    </style></head><body>
    <h1>🔴 Sophie — Monitor Conversazioni</h1>
    <p style="color:#888">""" + str(len(sessioni)) + """ sessioni — aggiorna la pagina per vedere le nuove</p>
    """
    for sid, messaggi in sessioni.items():
        ts_inizio = messaggi[-1][2].strftime("%d/%m/%Y %H:%M") if messaggi else ""
        html += f'<div class="sessione"><h3>Sessione: {sid[:12]}... — {ts_inizio}</h3><div class="wrap">'
        for ruolo, testo, ts in reversed(messaggi):
            classe = "cliente" if ruolo == "cliente" else "sophie"
            html += f'<div class="msg {classe}">{testo}<div class="ts">{ts.strftime("%H:%M:%S")}</div></div>'
        html += '</div></div>'
    html += "</body></html>"
    return html


if __name__ == "__main__":
    print("=" * 50)
    print("SOPHIE - Assistente virtuale Starpizza")
    print("powered by Arcanum v9")
    print("=" * 50)
    print("Apri: http://localhost:5000")
    print("CTRL+C per fermare")
    print("=" * 50)
    app.run(host="0.0.0.0", debug=False, port=5000)
