from flask import Flask, request, jsonify, render_template_string
import psycopg2
import anthropic
import json
import os
import re
from dotenv import load_dotenv
import numpy as np

load_dotenv()
DATABASE_URL  = os.getenv("DATABASE_URL")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    """Genera embedding OpenAI per un testo."""
    try:
        import urllib.request
        import json as json_lib
        data = json_lib.dumps({
            "input": text[:500],
            "model": "text-embedding-3-small"
        }).encode('utf-8')
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json_lib.loads(resp.read())
            return result["data"][0]["embedding"]
    except Exception as e:
        print(f"Errore embedding: {e}")
        return None

def cosine_similarity(a, b):
    """Calcola similarita coseno tra due vettori."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
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
body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; color: #222; display: flex; flex-direction: column; height: 100vh; font-size: 16px; }
header { background: #c0392b; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.hlogo { display: flex; align-items: center; gap: 12px; }
.hav { width: 40px; height: 40px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #c0392b; font-size: 1.2rem; }
.hname h1 { font-size: 1.15rem; margin: 0; color: white; }
.hname p { font-size: 0.82rem; color: rgba(255,255,255,0.85); margin: 0; }
.hright { display: flex; align-items: center; gap: 8px; }
.hright span { font-size: 0.75rem; color: rgba(255,255,255,0.6); }
#rbtn { background: none; border: 1px solid rgba(255,255,255,0.4); color: rgba(255,255,255,0.8); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
#rbtn:hover { border-color: white; color: white; }
#chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; background: #f9f9f9; }
.row { display: flex; align-items: flex-end; gap: 8px; }
.row.user { flex-direction: row-reverse; }
.mav { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; font-weight: bold; }
.mav.s { background: #c0392b; color: white; }
.mav.u { background: #ddd; color: #666; }
.bub { max-width: 72%; padding: 12px 16px; border-radius: 16px; font-size: 1rem; line-height: 1.65; white-space: pre-wrap; }
.bub.s { background: white; border: 1px solid #e0e0e0; border-bottom-left-radius: 4px; color: #222; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.bub.u { background: #c0392b; color: white; border-bottom-right-radius: 4px; }
.bub.loading { color: #aaa; font-style: italic; background: white; border: 1px solid #e0e0e0; }
.src { margin-top: 7px; padding-top: 7px; border-top: 1px solid #eee; font-size: 0.75rem; color: #aaa; display: flex; flex-wrap: wrap; gap: 4px; }
.tag { display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 0.72rem; }
.td { background: #eafaea; color: #3a8a3a; }
.te { background: #eaf0fa; color: #3a5aaa; }
.prow { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.pcard { background: #fff5f5; border: 1px solid #f0c0c0; border-radius: 8px; padding: 8px 13px; font-size: 0.85rem; text-decoration: none; color: #c0392b; display: inline-block; transition: all 0.2s; }
.pcard:hover { background: #c0392b; color: white; border-color: #c0392b; }
.pcard strong { display: block; color: #333; font-size: 0.88rem; margin-top: 2px; }
.pcard:hover strong { color: white; }
#foot { padding: 12px 16px; background: white; border-top: 1px solid #eee; display: flex; gap: 10px; align-items: center; }
#inp { flex: 1; background: #f5f5f5; border: 1.5px solid #ddd; color: #222; padding: 11px 18px; border-radius: 24px; font-size: 1rem; outline: none; transition: border 0.2s; }
#inp:focus { border-color: #c0392b; background: white; }
#inp::placeholder { color: #bbb; }
#sbtn { background: #c0392b; color: white; border: none; width: 44px; height: 44px; border-radius: 50%; cursor: pointer; font-size: 1.2rem; flex-shrink: 0; transition: background 0.2s; }
#sbtn:hover { background: #a93226; }
#sbtn:disabled { opacity: 0.35; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="hlogo">
    <div class="hav">S</div>
    <div class="hname"><h1>Sophie</h1><p>Assistente virtuale Starpizza &mdash; Online</p></div>
  </div>
  <div class="hright">
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
  escaped = escaped.replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank" style="color:#c0392b;font-weight:600;text-decoration:underline;">&#128279; Apri link</a>');
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
    return jsonify({"text": "Ciao! Sono Sophie, l'assistente virtuale di Starpizza 👋<br>Sto migliorando ogni giorno grazie alle richieste dei clienti, quindi più mi chiedi, più divento precisa 😊<br>Come posso aiutarti?"})


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

    # Recupera correzioni simili salvate dal titolare (autoapprendimento)
    correzioni_ctx = ""
    try:
        conn_c = psycopg2.connect(DATABASE_URL)
        cur_c  = conn_c.cursor()
        cur_c.execute("""
            CREATE TABLE IF NOT EXISTS correzioni (
                id SERIAL PRIMARY KEY, domanda_cliente TEXT,
                risposta_sophie TEXT, risposta_corretta TEXT,
                creato_il TIMESTAMP DEFAULT NOW()
            )
        """)
        # Ricerca semantica con embeddings OpenAI
        rows_c = []
        msg_embedding = get_embedding(message) if OPENAI_API_KEY else None

        if msg_embedding:
            # Prendi tutte le correzioni con embedding (se disponibile)
            cur_c.execute("""
                SELECT domanda_cliente, risposta_corretta, embedding
                FROM correzioni
                ORDER BY creato_il DESC
            """)
            all_corrections = cur_c.fetchall()

            if all_corrections:
                scored = []
                for dom, risp, emb_json in all_corrections:
                    if emb_json:
                        try:
                            emb = json.loads(emb_json)
                            score = cosine_similarity(msg_embedding, emb)
                            scored.append((score, dom, risp))
                        except:
                            pass
                    else:
                        # Correzioni senza embedding: usa keyword fallback
                        if any(w.lower() in (dom or '').lower() for w in message.split() if len(w) > 2):
                            scored.append((0.5, dom, risp))

                scored.sort(reverse=True)
                rows_c = [(dom, risp) for score, dom, risp in scored[:3] if score > 0.3]

        # Fallback keyword se no OpenAI o nessun risultato semantico
        if not rows_c:
            words_c = [w.lower() for w in message.split() if len(w) > 2][:8]
            if words_c:
                like_c = " OR ".join(["LOWER(domanda_cliente) LIKE %s"] * len(words_c))
                params_c = [f"%{w}%" for w in words_c]
                cur_c.execute(f"""
                    SELECT domanda_cliente, risposta_corretta
                    FROM correzioni WHERE {like_c}
                    ORDER BY creato_il DESC LIMIT 3
                """, params_c)
                rows_c = cur_c.fetchall()
            if rows_c:
                correzioni_ctx = "\n\n=== RISPOSTE CORRETTE DAL TEAM STARPIZZA (usa come riferimento) ===\n"
                for dom, risp in rows_c:
                    correzioni_ctx += f"\nDomanda: {dom[:300]}\nRisposta ideale: {risp[:1000]}\n"
        conn_c.commit()
        cur_c.close(); conn_c.close()
    except Exception as e:
        print(f"Errore correzioni: {e}")

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

        "Sei Sophie, assistente virtuale professionale di Starpizza.\n\n"
        "ESEMPI DI RISPOSTE IDEALI (segui questo stile esatto):\n"
        "---\n"
        "Domanda cliente: cella lievitazione per 4 carrelli 60x80\n"
        "Risposta ideale: Per una cella lievitazione per 4 carrelli 60x80 e possibile realizzarla con 2 porte con dimensioni esterne 1900x2000 mm oppure ad una porta con dimensioni 1000x3600. Puoi selezionarla alla pagina: https://starpizza.org/negozio/cella-di-lievitazione/\n"
        "---\n"
        "Nota: le risposte ideali sono brevi, dirette, con dimensioni precise e link. Segui questo modello.\n\n"
        "PERSONALITA:\n"
        "- BREVITA ASSOLUTA: massimo 2-3 frasi. Stop. Non aggiungere mai spiegazioni extra.\n"
        "- UNA sola domanda di follow-up al massimo.\n"
        "- Rispondi SEMPRE nella lingua del cliente.\n\n"
        "REGOLE:\n"
        f"- {note}{prod_note}\n"
        "- NON inventare prezzi o specifiche tecniche.\n"
        "- NON citare mai il brand del produttore.\n"
        "- LINK: usa SOLO i link dalla sezione PRODOTTI STARPIZZA TROVATI qui sotto. NON inventare mai link. Se non trovi il prodotto nei PRODOTTI TROVATI, manda il cliente su starpizza.org/negozio senza inventare URL.\n"
        "- EMAIL: chiedi solo per preventivi, resi o assistenza.\n"
        + docs_ctx + email_ctx + products_ctx + correzioni_ctx
    )
    history.append({"role": "user", "content": message})

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=180,
            temperature=0.1,
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
            CREATE TABLE IF NOT EXISTS correzioni (
                id SERIAL PRIMARY KEY, domanda_cliente TEXT,
                risposta_sophie TEXT, risposta_corretta TEXT,
                creato_il TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.execute("SELECT id, session_id, ruolo, testo, creato_il FROM chats ORDER BY creato_il DESC LIMIT 300")
        righe = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM correzioni")
        n_correzioni = cur.fetchone()[0]
        cur.close(); conn.close()
    except Exception as e:
        return f"Errore DB: {e}"

    sessioni = {}
    for mid, sid, ruolo, testo, ts in righe:
        if sid not in sessioni:
            sessioni[sid] = []
        sessioni[sid].append((mid, ruolo, testo, ts))

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Sophie Admin</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
.topbar { background: #c0392b; color: white; padding: 14px 24px; display: flex; justify-content: space-between; }
.topbar h1 { font-size: 1.2rem; }
.container { max-width: 900px; margin: 24px auto; padding: 0 16px; }
.stats { display: flex; gap: 16px; margin-bottom: 24px; }
.stat { background: white; border-radius: 10px; padding: 14px 20px; flex: 1; box-shadow: 0 2px 6px rgba(0,0,0,0.07); }
.stat h3 { font-size: 1.6rem; color: #c0392b; }
.stat p { font-size: 0.8rem; color: #888; margin-top: 4px; }
.sessione { background: white; border-radius: 12px; padding: 18px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.sess-header { font-size: 0.8rem; color: #aaa; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #eee; }
.wrap { display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 10px 14px; border-radius: 10px; max-width: 80%; font-size: 0.95rem; line-height: 1.5; }
.cliente { background: #c0392b; color: white; align-self: flex-end; }
.sophie { background: #f0f0f0; color: #222; align-self: flex-start; }
.ts { font-size: 0.7rem; opacity: 0.6; margin-top: 3px; }
.intervieni { margin-top: 16px; border-top: 1px solid #eee; padding-top: 14px; }
.intervieni label { font-size: 0.85rem; color: #555; font-weight: 600; display: block; margin-bottom: 6px; }
.intervieni textarea { width: 100%; border: 1.5px solid #ddd; border-radius: 8px; padding: 10px; font-size: 0.95rem; resize: vertical; min-height: 80px; font-family: inherit; }
.btn-correggi { background: #c0392b; color: white; border: none; padding: 9px 20px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; margin-top: 8px; }
.refresh { font-size: 0.8rem; color: #888; text-align: center; margin-top: 20px; }
</style></head><body>
<div class="topbar"><h1>🔴 Sophie — Pannello Controllo</h1><span>""" + str(len(sessioni)) + """ sessioni | """ + str(n_correzioni) + """ correzioni</span></div>
<div class="container">
<div class="stats">
<div class="stat"><h3>""" + str(len(sessioni)) + """</h3><p>Conversazioni</p></div>
<div class="stat"><h3>""" + str(sum(len(v) for v in sessioni.values())) + """</h3><p>Messaggi</p></div>
<div class="stat"><h3>""" + str(n_correzioni) + """</h3><p>Correzioni</p></div>
</div>
"""
    for sid, messaggi in sessioni.items():
        ts_inizio = messaggi[-1][3].strftime("%d/%m/%Y %H:%M") if messaggi else ""
        ultima_risposta = ""
        ultima_domanda = ""
        ultimo_id = None
        for mid, ruolo, testo, ts in messaggi:
            if ruolo == "sophie" and ultimo_id is None:
                ultima_risposta = testo
                ultimo_id = mid
            if ruolo == "cliente" and not ultima_domanda:
                ultima_domanda = testo

        html += f'<div class="sessione"><div class="sess-header">Sessione {sid[:16]}... — {ts_inizio}</div><div class="wrap">'
        for mid, ruolo, testo, ts in reversed(messaggi):
            classe = "cliente" if ruolo == "cliente" else "sophie"
            html += f'<div class="msg {classe}">{testo}<div class="ts">{ts.strftime("%H:%M")}</div></div>'

        corr_id = str(ultimo_id) if ultimo_id else "none"
        dom_safe = ultima_domanda.replace('"', '&quot;')
        risp_safe = ultima_risposta.replace('"', '&quot;').replace('<','&lt;').replace('>','&gt;')
        html += '</div>'
        html += '<div class="intervieni"><label>✏️ Correggi ultima risposta Sophie:</label>'
        html += f'<textarea id="corr_{corr_id}">{risp_safe}</textarea>'
        html += f'<input type="hidden" id="dom_{corr_id}" value="{dom_safe}">'
        html += f'<input type="hidden" id="orig_{corr_id}" value="{risp_safe}">'
        html += f'<button class="btn-correggi" onclick="salvaCorrezione(\'{corr_id}\')">💾 Salva correzione</button>'
        html += f'<span id="ok_{corr_id}" style="display:none;color:#27ae60;margin-left:8px;">✅ Salvata!</span>'
        html += '</div></div>'

    html += """<p class="refresh">🔄 <a href="/admin/chat">Aggiorna</a></p></div>
<script>
function salvaCorrezione(msgId) {
  var textarea = document.getElementById('corr_' + msgId);
  var testo = textarea.value.trim();
  var domanda = document.getElementById('dom_' + msgId) ? document.getElementById('dom_' + msgId).value : '';
  var originale = document.getElementById('orig_' + msgId) ? document.getElementById('orig_' + msgId).value : '';
  if (!testo) { alert('Scrivi una correzione!'); return; }
  fetch('/admin/correggi', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({correzione: testo, domanda: domanda, risposta_originale: originale})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      textarea.style.border = '2px solid #27ae60';
      textarea.style.background = '#f0fff0';
      document.getElementById('ok_' + msgId).style.display = 'inline';
    } else { alert('Errore: ' + (d.errore || 'riprova')); }
  }).catch(function() { alert('Errore connessione'); });
}
</script></body></html>"""
    return html


@app.route("/admin/correggi", methods=["POST"])
def admin_correggi():
    data = request.json or {}
    correzione = data.get("correzione", "").strip()
    domanda = data.get("domanda", "").strip()
    risposta_originale = data.get("risposta_originale", "").strip()
    if not correzione:
        return jsonify({"ok": False, "errore": "correzione vuota"})
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS correzioni (
                id SERIAL PRIMARY KEY, domanda_cliente TEXT,
                risposta_sophie TEXT, risposta_corretta TEXT,
                creato_il TIMESTAMP DEFAULT NOW(),
                embedding TEXT
            )
        """)
        # Calcola embedding automaticamente
        emb_json = None
        if OPENAI_API_KEY and domanda:
            try:
                emb = get_embedding(domanda)
                if emb:
                    emb_json = json.dumps(emb)
            except Exception as emb_err:
                print(f"Embedding non calcolato: {emb_err}")

        cur.execute("""
            INSERT INTO correzioni (domanda_cliente, risposta_sophie, risposta_corretta, embedding)
            VALUES (%s, %s, %s, %s)
        """, (domanda, risposta_originale, correzione, emb_json))
        conn.commit()
        cur.close(); conn.close()
        print(f"Correzione salvata{'+ embedding' if emb_json else ''}: {domanda[:50]}")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"Errore correzione: {e}")
        return jsonify({"ok": False, "errore": str(e)})
