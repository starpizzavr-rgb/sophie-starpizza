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
        words_c = [w.lower() for w in message.split() if len(w) > 4][:5]
        if words_c:
            like_c = " OR ".join(["LOWER(domanda_cliente) LIKE %s"] * len(words_c))
            params_c = [f"%{w}%" for w in words_c]
            cur_c.execute(f"""
                SELECT domanda_cliente, risposta_corretta
                FROM correzioni
                WHERE {like_c}
                ORDER BY creato_il DESC LIMIT 3
            """, params_c)
            rows_c = cur_c.fetchall()
            if rows_c:
                correzioni_ctx = "\n\n=== RISPOSTE CORRETTE DAL TEAM STARPIZZA (usa come riferimento) ===\n"
                for dom, risp in rows_c:
                    correzioni_ctx += f"\nDomanda: {dom[:200]}\nRisposta ideale: {risp[:400]}\n"
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
        "REGOLE IMPORTANTI:\n"
        f"- {note}{prod_note}\n"
        "- NON inventare prezzi, dimensioni o specifiche tecniche\n"
        "- NON menzionare mai il nome del produttore o brand\n"
        "- Per prezzi rimanda al sito o al team Starpizza\n"
        "- RACCOLTA EMAIL: chiedi la email del cliente SOLO quando ha senso: "
        "quando chiede un preventivo, vuole essere ricontattato, segnala un problema, un reso o un reclamo. "
        "Esempio: Posso farti richiamare dal nostro team, mi lasci la sua email? "
        "Non chiedere la email se il cliente cerca solo info generali. "
        "Una volta raccolta, NON chiederla di nuovo.\n"
        "- TRASPARENZA: puoi dire al cliente che sei un assistente virtuale in fase di apprendimento "
        "e che le conversazioni vengono seguite dal team Starpizza per migliorare il servizio.\n"
        + docs_ctx + email_ctx + products_ctx + correzioni_ctx
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
    """Pagina privata per monitorare le conversazioni e intervenire."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        # Crea tabelle se non esistono
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY, session_id VARCHAR(64),
                ruolo VARCHAR(16), testo TEXT, creato_il TIMESTAMP DEFAULT NOW(),
                corretto BOOLEAN DEFAULT FALSE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS correzioni (
                id SERIAL PRIMARY KEY,
                domanda_cliente TEXT,
                risposta_sophie TEXT,
                risposta_corretta TEXT,
                creato_il TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.execute("""
            SELECT id, session_id, ruolo, testo, creato_il
            FROM chats ORDER BY creato_il DESC LIMIT 300
        """)
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

    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Sophie Admin</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
.topbar { background: #c0392b; color: white; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; }
.topbar h1 { font-size: 1.2rem; }
.topbar span { font-size: 0.85rem; opacity: 0.85; }
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
.intervieni textarea:focus { border-color: #c0392b; outline: none; }
.btn-correggi { background: #c0392b; color: white; border: none; padding: 9px 20px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; margin-top: 8px; }
.btn-correggi:hover { background: #a93226; }
.ok { color: #27ae60; font-size: 0.85rem; margin-left: 10px; display: none; }
.refresh { font-size: 0.8rem; color: #888; text-align: center; margin-top: 20px; }
</style>
</head><body>
<div class="topbar">
  <h1>🔴 Sophie — Pannello di Controllo</h1>
  <span>""" + str(len(sessioni)) + """ sessioni &nbsp;|&nbsp; """ + str(n_correzioni) + """ correzioni salvate</span>
</div>
<div class="container">
  <div class="stats">
    <div class="stat"><h3>""" + str(len(sessioni)) + """</h3><p>Conversazioni totali</p></div>
    <div class="stat"><h3>""" + str(sum(len(v) for v in sessioni.values())) + """</h3><p>Messaggi totali</p></div>
    <div class="stat"><h3>""" + str(n_correzioni) + """</h3><p>Correzioni salvate</p></div>
  </div>
"""
    for sid, messaggi in sessioni.items():
        ts_inizio = messaggi[-1][3].strftime("%d/%m/%Y %H:%M") if messaggi else ""
        html += f'''<div class="sessione">
  <div class="sess-header">Sessione {sid[:16]}... &nbsp;—&nbsp; {ts_inizio}</div>
  <div class="wrap">'''
        ultima_domanda = ""
        ultima_risposta = ""
        ultimo_id = None
        for mid, ruolo, testo, ts in reversed(messaggi):
            classe = "cliente" if ruolo == "cliente" else "sophie"
            html += f'<div class="msg {classe}">{testo}<div class="ts">{ts.strftime("%H:%M")}</div></div>'
            if ruolo == "cliente":
                ultima_domanda = testo
            if ruolo == "sophie":
                ultima_risposta = testo
                ultimo_id = mid

        # Box intervento umano
        html += f'''</div>
  <div class="intervieni">
    <label>✏️ Correggi l'ultima risposta di Sophie (Sophie imparerà per il futuro):</label>
    <textarea id="corr_{ultimo_id}" placeholder="Scrivi qui la risposta migliore...">{ultima_risposta}</textarea>
    <button class="btn-correggi" onclick="salvaCorrezione({ultimo_id}, '{sid}')">💾 Salva correzione</button>
    <span class="ok" id="ok_{ultimo_id}">✅ Salvata!</span>
  </div>
</div>'''

    html += """
<p class="refresh">🔄 <a href="/admin/chat">Aggiorna pagina</a> per vedere nuove conversazioni</p>
</div>
<script>
function salvaCorrezione(msgId, sid) {
  var testo = document.getElementById('corr_' + msgId).value;
  fetch('/admin/correggi', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({msg_id: msgId, correzione: testo, session_id: sid})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      document.getElementById('ok_' + msgId).style.display = 'inline';
      setTimeout(function() { document.getElementById('ok_' + msgId).style.display = 'none'; }, 3000);
    }
  });
}
</script>
</body></html>"""
    return html


@app.route("/admin/correggi", methods=["POST"])
def admin_correggi():
    """Salva la correzione umana e la usa come esempio per Sophie."""
    data       = request.json or {}
    msg_id     = data.get("msg_id")
    correzione = data.get("correzione", "").strip()
    session_id = data.get("session_id", "")

    if not correzione:
        return jsonify({"ok": False})

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()

        # Trova la domanda del cliente nella stessa sessione
        cur.execute("""
            SELECT testo FROM chats
            WHERE session_id = %s AND ruolo = 'cliente'
            ORDER BY creato_il DESC LIMIT 1
        """, (session_id,))
        row = cur.fetchone()
        domanda = row[0] if row else ""

        # Trova la risposta originale di Sophie
        cur.execute("SELECT testo FROM chats WHERE id = %s", (msg_id,))
        row2 = cur.fetchone()
        risposta_originale = row2[0] if row2 else ""

        # Salva nella tabella correzioni
        cur.execute("""
            INSERT INTO correzioni (domanda_cliente, risposta_sophie, risposta_corretta)
            VALUES (%s, %s, %s)
        """, (domanda, risposta_originale, correzione))

        conn.commit()
        cur.close(); conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        print(f"Errore correzione: {e}")
        return jsonify({"ok": False})


if __name__ == "__main__":
    print("=" * 50)
    print("SOPHIE - Assistente virtuale Starpizza")
    print("powered by Arcanum v9")
    print("=" * 50)
    print("Apri: http://localhost:5000")
    print("CTRL+C per fermare")
    print("=" * 50)
    app.run(host="0.0.0.0", debug=False, port=5000)
