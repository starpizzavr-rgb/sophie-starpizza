from flask import Flask, request, jsonify, render_template_string
import psycopg2
import anthropic
import json
import os
import re
import threading
import time
from dotenv import load_dotenv
import numpy as np

load_dotenv()
DATABASE_URL   = os.getenv("DATABASE_URL")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# GMAIL — Invio e lettura email per Sophie
# ============================================================

SOPHIE_EMAIL       = os.getenv("SOPHIE_EMAIL", "sophie@starpizza.org")
GMAIL_CLIENT_ID    = os.getenv("GOOGLE_CLIENT_ID")
GMAIL_CLIENT_SECRET= os.getenv("GOOGLE_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN= os.getenv("GOOGLE_REFRESH_TOKEN")

def gmail_get_access_token():
    """Ottiene access token Gmail usando il refresh token."""
    if not all([GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN]):
        return None
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({
            "client_id":     GMAIL_CLIENT_ID,
            "client_secret": GMAIL_CLIENT_SECRET,
            "refresh_token": GMAIL_REFRESH_TOKEN,
            "grant_type":    "refresh_token"
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("access_token")
    except Exception as e:
        print(f"Gmail token error: {e}")
        return None

def gmail_invia(destinatario, oggetto, corpo_html, cc=None):
    """Invia una email tramite Gmail API."""
    import base64, urllib.request
    token = gmail_get_access_token()
    if not token:
        print("Gmail: token non disponibile")
        return False
    try:
        cc_line = f"Cc: {cc}\r\n" if cc else ""
        raw_email = (
            f"From: Sophie Starpizza <{SOPHIE_EMAIL}>\r\n"
            f"To: {destinatario}\r\n"
            f"{cc_line}"
            f"Subject: {oggetto}\r\n"
            f"MIME-Version: 1.0\r\n"
            f"Content-Type: text/html; charset=utf-8\r\n"
            f"\r\n"
            f"{corpo_html}"
        )
        raw_b64 = base64.urlsafe_b64encode(raw_email.encode("utf-8")).decode("utf-8")
        body = json.dumps({"raw": raw_b64}).encode("utf-8")
        req = urllib.request.Request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Gmail: email inviata a {destinatario}")
            return True
    except Exception as e:
        print(f"Gmail invio error: {e}")
        return False

def gmail_leggi_non_lette(max_results=10):
    """Legge le email non lette nella casella Sophie."""
    import urllib.request, urllib.parse, base64
    token = gmail_get_access_token()
    if not token:
        return []
    try:
        # Cerca email non lette
        params = urllib.parse.urlencode({
            "q": "is:unread -from:me",
            "maxResults": str(max_results)
        })
        req = urllib.request.Request(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages?{params}",
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            messages = result.get("messages", [])

        emails = []
        for msg in messages:
            msg_id = msg["id"]
            req2 = urllib.request.Request(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}?format=full",
                headers={"Authorization": f"Bearer {token}"}
            )
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                full_msg = json.loads(resp2.read())

            headers = {h["name"]: h["value"] for h in full_msg.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "(nessun oggetto)")
            sender  = headers.get("From", "")
            reply_to= headers.get("Reply-To", sender)

            # Estrai corpo
            corpo = ""
            payload = full_msg.get("payload", {})
            if payload.get("body", {}).get("data"):
                corpo = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            elif payload.get("parts"):
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                        corpo = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                        break

            emails.append({
                "id":       msg_id,
                "subject":  subject,
                "sender":   sender,
                "reply_to": reply_to,
                "body":     corpo[:3000]
            })
        return emails
    except Exception as e:
        print(f"Gmail lettura error: {e}")
        return []

def gmail_segna_letta(msg_id):
    """Segna un messaggio come letto."""
    import urllib.request
    token = gmail_get_access_token()
    if not token:
        return
    try:
        body = json.dumps({"removeLabelIds": ["UNREAD"]}).encode("utf-8")
        req = urllib.request.Request(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}/modify",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Gmail segna letta error: {e}")

def sophie_rispondi_email(email):
    """Sophie legge un'email e genera una risposta automatica."""
    try:
        prompt = (
            f"Sei Sophie, assistente virtuale di Starpizza & co SRL (starpizza.org).\n"
            f"Hai ricevuto questa email:\n"
            f"Da: {email['sender']}\n"
            f"Oggetto: {email['subject']}\n"
            f"Testo: {email['body']}\n\n"
            f"Rispondi in modo professionale e cordiale, nella lingua del mittente.\n\n"
            f"=== CONDIZIONI DI VENDITA STARPIZZA ===\n"
            f"PAGAMENTO — metodi accettati:\n"
            f"1. BONIFICO BANCARIO anticipato entro 24 ore dalla conferma ordine (30% all'ordine, saldo a merce pronta)\n"
            f"   - IBAN Italia: IT77S3609201600428248836096 — Intestatario: Starpizza & co SRL — Banca: Qonto\n"
            f"   - IBAN Estero: IT43 D060 4511 7000 0000 5004 189 — BIC SWIFT: CRBZIT2B096 — Banca: Cassa di Risparmio di Bolzano\n"
            f"   - Inviare copia bonifico con CRO a: starpizzavr@gmail.com\n"
            f"2. CONTRASSEGNO (pagamento alla consegna)\n"
            f"3. CARTA DI CREDITO\n"
            f"4. PAYPAL — con PayPal è possibile pagare anche in 3 rate senza interessi per importi fino a 2.000 euro\n\n"
            f"SPEDIZIONE:\n"
            f"- Spese di trasporto variabili in base a destinazione, peso e dimensioni\n"
            f"- Consegna al piano, facchinaggio e sponda idraulica disponibili su richiesta (costo extra)\n"
            f"- Comunicare eventuali zone disagiate prima dell'ordine\n"
            f"- All'arrivo controllare l'imballaggio: se danneggiato accettare con RISERVA DI CONTROLLO\n\n"
            f"GARANZIA:\n"
            f"- 1 anno dalla consegna contro difetti di materiale o fabbricazione\n"
            f"- Spese di reso a carico del cliente\n\n"
            f"RECESSO:\n"
            f"- Senza acconto: pratica chiusa automaticamente\n"
            f"- Con acconto versato: nessun rimborso, buono acquisto valido per l'anno in corso\n"
            f"- Diritto di recesso solo per consumatori finali (non partite IVA), entro 14 giorni dalla consegna\n\n"
            f"INSTALLAZIONE: esclusa, preventivo disponibile su richiesta\n\n"
            f"=== ISTRUZIONI PER SOPHIE ===\n"
            f"- Se chiedono metodo di pagamento: bonifico bancario, contrassegno, carta di credito, PayPal (con PayPal anche 3 rate per importi fino a 2.000€)\n"
            f"- Se chiedono contrassegno: SÌ, è accettato\n"
            f"- Se chiedono rate o finanziamento: disponibile con PayPal in 3 rate senza interessi fino a 2.000€\n"
            f"- Se chiedono IBAN o coordinate bancarie: fornisci SOLO quelle indicate sopra\n"
            f"- Se chiedono spedizione o tracking: chiedi ragione sociale o email per verificare\n"
            f"- Se chiedono fattura: viene emessa automaticamente e inviata via email\n"
            f"- Se chiedono installazione: non è inclusa ma disponibile su preventivo\n"
            f"- Se chiedono garanzia: 1 anno dalla consegna\n"
            f"- NON inventare prezzi, dati o informazioni non presenti\n"
            f"- Firma sempre come: Sophie | Assistente Virtuale Starpizza | starpizza.org\n"
            f"Rispondi SOLO con il testo dell'email, senza oggetto."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"Sophie risposta email error: {e}")
        return None

def email_gia_elaborata(msg_id):
    """Controlla se abbiamo già elaborato questa email."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_elaborate (
                id SERIAL PRIMARY KEY,
                msg_id VARCHAR(128) UNIQUE,
                elaborata_il TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.execute("SELECT id FROM email_elaborate WHERE msg_id=%s", (msg_id,))
        exists = cur.fetchone() is not None
        cur.close(); conn.close()
        return exists
    except:
        return False

def segna_email_elaborata(msg_id):
    """Registra l'email come elaborata nel DB."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO email_elaborate (msg_id) VALUES (%s)
            ON CONFLICT (msg_id) DO NOTHING
        """, (msg_id,))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"Segna email elaborata error: {e}")

def loop_email():
    """
    Thread in background: ogni 5 minuti legge le email non lette
    e risponde automaticamente.
    """
    print("Sophie Email Loop: avviato")
    time.sleep(30)  # aspetta avvio completo
    while True:
        try:
            emails = gmail_leggi_non_lette(max_results=10)
            for email in emails:
                # Controllo doppio: DB + segna letta Gmail
                if email_gia_elaborata(email["id"]):
                    continue
                print(f"Sophie Email: elaboro email da {email['sender']} — {email['subject']}")
                risposta = sophie_rispondi_email(email)
                if risposta:
                    import re as _re
                    email_match = _re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', email['reply_to'])
                    if email_match:
                        dest = email_match.group(0)
                        oggetto = f"Re: {email['subject']}"
                        corpo_html = risposta.replace("\n", "<br>")
                        if gmail_invia(dest, oggetto, corpo_html):
                            segna_email_elaborata(email["id"])
                            gmail_segna_letta(email["id"])
                            print(f"Sophie Email: risposto a {dest}")
        except Exception as e:
            print(f"Loop email error: {e}")
        time.sleep(300)  # ogni 5 minuti

# ============================================================
# NOTIFICHE SPEDIZIONE — Monitor SpediamoPro
# ============================================================

# Tabella per tracciare spedizioni già notificate
def init_notifiche_db():
    """Crea la tabella notifiche se non esiste."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS spedizioni_notificate (
                id              SERIAL PRIMARY KEY,
                shipment_code   VARCHAR(64) UNIQUE,
                stato_notificato INTEGER,
                email_cliente   VARCHAR(256),
                nome_cliente    VARCHAR(256),
                notificato_il   TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"Init notifiche DB error: {e}")

def notifica_gia_inviata(shipment_code, stato):
    """Controlla se abbiamo già notificato questo stato per questa spedizione."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute(
            "SELECT id FROM spedizioni_notificate WHERE shipment_code=%s AND stato_notificato=%s",
            (shipment_code, stato)
        )
        exists = cur.fetchone() is not None
        cur.close(); conn.close()
        return exists
    except:
        return False

def segna_notifica_inviata(shipment_code, stato, email_cliente, nome_cliente):
    """Registra che abbiamo inviato la notifica."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO spedizioni_notificate (shipment_code, stato_notificato, email_cliente, nome_cliente)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (shipment_code) DO UPDATE SET
                stato_notificato = EXCLUDED.stato_notificato,
                notificato_il    = NOW()
        """, (shipment_code, stato, email_cliente, nome_cliente))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"Segna notifica error: {e}")

def genera_email_spedizione(nome, tracking_url, tracking_code, corriere, data_consegna, stato):
    """Genera il corpo HTML dell'email di notifica spedizione."""
    if stato in [6, 7, 8]:  # spedita / in transito
        titolo  = "Il tuo ordine è in viaggio! 🚚"
        intro   = f"Ciao {nome},<br><br>ottima notizia! Il tuo ordine è stato spedito ed è in viaggio verso di te."
        colore  = "#27ae60"
    elif stato == 9:  # in consegna oggi
        titolo  = "Il tuo ordine arriva oggi! 📦"
        intro   = f"Ciao {nome},<br><br>oggi è il grande giorno! Il corriere consegnerà il tuo ordine in giornata."
        colore  = "#e67e22"
    elif stato == 11:  # eccezione/problema
        titolo  = "Aggiornamento sulla tua spedizione ⚠️"
        intro   = f"Ciao {nome},<br><br>volevo aggiornarti sulla tua spedizione. Si è verificato un piccolo intoppo."
        colore  = "#e74c3c"
    elif stato == 12:  # giacenza
        titolo  = "Il tuo pacco è in giacenza 📬"
        intro   = f"Ciao {nome},<br><br>il corriere ha tentato la consegna ma non ha trovato nessuno. Il tuo pacco è ora in giacenza."
        colore  = "#e74c3c"
    else:
        return None

    data_str = f"<br><strong>Consegna prevista:</strong> {data_consegna}" if data_consegna else ""
    link_str = f'<br><br><a href="{tracking_url}" style="background:{colore};color:white;padding:12px 24px;border-radius:6px;text-decoration:none;display:inline-block;margin-top:8px;">📍 Traccia la spedizione</a>' if tracking_url else ""

    return f"""
    <div style="font-family:'Segoe UI',sans-serif;max-width:600px;margin:0 auto;background:#f9f9f9;border-radius:12px;overflow:hidden;">
        <div style="background:{colore};padding:24px;text-align:center;">
            <h1 style="color:white;margin:0;font-size:1.4rem;">{titolo}</h1>
        </div>
        <div style="padding:28px;background:white;">
            <p style="color:#333;line-height:1.7;">{intro}</p>
            <div style="background:#f5f5f5;border-radius:8px;padding:16px;margin:16px 0;">
                <strong>Corriere:</strong> {corriere}<br>
                <strong>Codice tracking:</strong> {tracking_code}
                {data_str}
            </div>
            {link_str}
            <p style="color:#888;font-size:0.85rem;margin-top:24px;">
                Per qualsiasi domanda rispondi a questa email o visita <a href="https://starpizza.org">starpizza.org</a>
            </p>
        </div>
        <div style="background:#c0392b;padding:14px;text-align:center;">
            <span style="color:white;font-size:0.8rem;">Sophie | Assistente Virtuale Starpizza</span>
        </div>
    </div>
    """

def loop_notifiche_spedizione():
    """
    Thread in background: ogni 15 minuti controlla le spedizioni
    e invia notifiche email ai clienti per stati importanti.
    """
    print("Sophie Notifiche: avviato")
    init_notifiche_db()
    time.sleep(60)  # aspetta avvio completo
    while True:
        try:
            token = spediamopro_get_token()
            if token:
                import urllib.request
                # Cerca spedizioni attive (stati 5-12)
                body = json.dumps({"statuses": [5, 6, 7, 8, 9, 11, 12]}).encode("utf-8")
                req  = urllib.request.Request(
                    f"{SPEDIAMOPRO_BASE_URL}/shipments/search?perPage=50",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type":  "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
                    spedizioni = result.get("data", [])

                STATI_DA_NOTIFICARE = {6, 7, 8, 9, 11, 12}
                from datetime import datetime, timedelta
                data_limite = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

                for sped in spedizioni:
                    # Salta spedizioni più vecchie di 30 giorni
                    data_creazione = sped.get("createdAt", sped.get("created_at", ""))
                    if data_creazione and data_creazione[:10] < data_limite:
                        continue
                    shipment_id   = sped.get("id")
                    shipment_code = sped.get("code", "")

                    # Recupera tracking dettagliato
                    tracking = spediamopro_get_tracking(shipment_id)
                    if not tracking:
                        continue

                    stato         = tracking.get("status", -1)
                    tracking_url  = tracking.get("url", "")
                    tracking_code = tracking.get("trackingCode", "")
                    corriere      = tracking.get("courier", "")
                    data_consegna = tracking.get("expectedDeliveryDate", "")

                    if stato not in STATI_DA_NOTIFICARE:
                        continue
                    if notifica_gia_inviata(shipment_code, stato):
                        continue

                    # Recupera email e nome destinatario dalla spedizione
                    consignee  = sped.get("consignee", {})
                    email_dest = consignee.get("email", "")
                    nome_dest  = consignee.get("name", "Cliente")

                    if not email_dest:
                        continue

                    corpo = genera_email_spedizione(
                        nome_dest, tracking_url, tracking_code,
                        corriere, data_consegna, stato
                    )
                    if not corpo:
                        continue

                    oggetti = {
                        6:  "Il tuo ordine Starpizza e in partenza",
                        7:  "Il tuo ordine Starpizza e in viaggio",
                        8:  "Il tuo ordine Starpizza e in transito",
                        9:  "Il tuo ordine Starpizza arriva oggi",
                        11: "Aggiornamento spedizione Starpizza",
                        12: "Il tuo pacco Starpizza e in giacenza",
                    }
                    oggetto = oggetti.get(stato, "Aggiornamento spedizione Starpizza")

                    if gmail_invia(email_dest, oggetto, corpo):
                        segna_notifica_inviata(shipment_code, stato, email_dest, nome_dest)
                        print(f"Notifica inviata: {shipment_code} stato {stato} → {email_dest}")

        except Exception as e:
            print(f"Loop notifiche error: {e}")

        time.sleep(900)  # ogni 15 minuti

# ============================================================
# SPEDIAMOPRO — Tracking spedizioni
# ============================================================

SPEDIAMOPRO_USERNAME  = os.getenv("SPEDIAMOPRO_USERNAME")   # email account SpediamoP ro
SPEDIAMOPRO_AUTHCODE  = os.getenv("SPEDIAMOPRO_AUTHCODE")   # authcode dal pannello Integrazioni
SPEDIAMOPRO_BASE_URL  = "https://core.spediamopro.com/api/v2"

# Cache token per evitare troppe chiamate auth
_spediamopro_token        = None
_spediamopro_token_expiry = 0

TRACKING_STATUS_MAP = {
    0:  "annullata",
    4:  "pagata e in attesa di elaborazione",
    5:  "elaborata — etichetta creata",
    6:  "ritiro richiesto",
    7:  "avviata — primo evento corriere",
    8:  "in transito",
    9:  "in consegna oggi",
    10: "consegnata ✅",
    11: "in eccezione (ritardo, mancata consegna o giacenza) — contattaci",
    12: "consegnata al punto di ritiro",
    13: "in attesa di elaborazione",
}

def spediamopro_get_token():
    """
    Ottiene il token per SpediamoP ro.
    Strategia 1: usa SPEDIAMOPRO_AUTHCODE direttamente come Bearer token.
    Strategia 2: se fallisce, tenta Basic Auth con username:authcode.
    """
    global _spediamopro_token, _spediamopro_token_expiry
    if not SPEDIAMOPRO_AUTHCODE:
        return None
    if _spediamopro_token and time.time() < _spediamopro_token_expiry - 60:
        return _spediamopro_token
    try:
        import urllib.request, urllib.parse, base64

        # Strategia 1: Authcode diretto come Bearer token (modalità più comune)
        # Verifica con una chiamata test al wallet endpoint
        test_req = urllib.request.Request(
            f"{SPEDIAMOPRO_BASE_URL}/wallet",
            headers={"Authorization": f"Bearer {SPEDIAMOPRO_AUTHCODE}"}
        )
        try:
            with urllib.request.urlopen(test_req, timeout=10) as resp:
                if resp.status == 200:
                    _spediamopro_token        = SPEDIAMOPRO_AUTHCODE
                    _spediamopro_token_expiry = time.time() + 86400  # 24h
                    print("SpediamoP ro: authcode usato direttamente come Bearer token")
                    return _spediamopro_token
        except Exception:
            pass

        # Strategia 2: Basic Auth username:authcode -> ottieni access_token
        if SPEDIAMOPRO_USERNAME:
            # Authcode come USERNAME, password VUOTA (doc SpediamoP ro)
            credenziali = base64.b64encode(
                f"{SPEDIAMOPRO_AUTHCODE}:".encode("utf-8")
            ).decode("utf-8")
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials",
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{SPEDIAMOPRO_BASE_URL}/auth/token",
                data=data,
                headers={
                    "Content-Type":  "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {credenziali}"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result                    = json.loads(resp.read())
                _spediamopro_token        = result.get("access_token")
                expires_in                = result.get("expires_in", 3600)
                _spediamopro_token_expiry = time.time() + expires_in
                print(f"SpediamoP ro token via Basic Auth, scade in {expires_in}s")
                return _spediamopro_token

    except Exception as e:
        print(f"SpediamoP ro auth error: {e}")
        return None

def spediamopro_cerca_spedizione(query_str):
    """
    Cerca una spedizione provando in sequenza:
    1. Direttamente per ID numerico -> GET /shipments/{id}
    2. Per codice alfanumerico -> GET /shipments/by-code/{code}
    3. Ricerca testuale -> POST /shipments/search
    """
    token = spediamopro_get_token()
    if not token:
        return None

    import urllib.request

    # Strategia 1: ID numerico puro
    query_clean = query_str.strip()
    if query_clean.isdigit():
        try:
            req = urllib.request.Request(
                f"{SPEDIAMOPRO_BASE_URL}/shipments/{query_clean}",
                headers={"Authorization": f"Bearer {token}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                data = result.get("data")
                if data:
                    print(f"SpediamoP ro: trovata per ID {query_clean}")
                    return data
        except Exception as e:
            print(f"SpediamoP ro by-id error: {e}")

    # Strategia 2: codice alfanumerico
    try:
        req = urllib.request.Request(
            f"{SPEDIAMOPRO_BASE_URL}/shipments/by-code/{query_clean}",
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            data = result.get("data")
            if data:
                print(f"SpediamoP ro: trovata per codice {query_clean}")
                return data
    except Exception as e:
        print(f"SpediamoP ro by-code error: {e}")

    # Strategia 3: ricerca testuale (email, nome, telefono)
    try:
        body = json.dumps({"search": query_str}).encode("utf-8")
        req  = urllib.request.Request(
            f"{SPEDIAMOPRO_BASE_URL}/shipments/search",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            items  = result.get("data", [])
            if items:
                print(f"SpediamoP ro: trovata per ricerca testuale '{query_str}'")
                return items[0]
            print(f"SpediamoP ro: nessun risultato per '{query_str}'")
            return None
    except Exception as e:
        print(f"SpediamoP ro search error: {e}")
        return None

def spediamopro_get_tracking(shipment_id):
    """Recupera il tracking dettagliato per ID spedizione."""
    token = spediamopro_get_token()
    if not token:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{SPEDIAMOPRO_BASE_URL}/shipments/{shipment_id}/tracking",
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("data")
    except Exception as e:
        print(f"SpediamoP ro tracking error: {e}")
        return None

def spediamopro_tracking_testo(query_str):
    """
    Funzione principale: cerca la spedizione e restituisce
    un testo pronto per Sophie con lo stato aggiornato.
    """
    spedizione = spediamopro_cerca_spedizione(query_str)
    if not spedizione:
        return None

    shipment_id   = spedizione.get("id")
    shipment_code = spedizione.get("code", "")

    tracking = spediamopro_get_tracking(shipment_id)
    if not tracking:
        return None

    status_code  = tracking.get("status", -1)
    status_label = TRACKING_STATUS_MAP.get(status_code, f"stato {status_code}")
    tracking_url = tracking.get("url", "")
    delivery_date= tracking.get("expectedDeliveryDate", "")
    corriere     = tracking.get("courier", "")
    tracking_code= tracking.get("trackingCode", "")

    # Ultimo evento
    events       = tracking.get("events", [])
    ultimo_evento= ""
    if events:
        ev           = events[-1]
        ultimo_evento= ev.get("description", "")

    testo = f"[TRACKING SPEDIZIONE]\n"
    testo += f"Codice spedizione: {shipment_code}\n"
    testo += f"Stato: {status_label}\n"
    if corriere:        testo += f"Corriere: {corriere}\n"
    if tracking_code:   testo += f"Codice tracking: {tracking_code}\n"
    if delivery_date:   testo += f"Data prevista consegna: {delivery_date}\n"
    if ultimo_evento:   testo += f"Ultimo aggiornamento: {ultimo_evento}\n"
    if tracking_url:    testo += f"Link tracking: {tracking_url}\n"

    return testo

# ============================================================
# GOOGLE DRIVE — indice cartelle + lettura PDF on-demand
# ============================================================

GOOGLE_CLIENT_ID       = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET   = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN   = os.getenv("GOOGLE_REFRESH_TOKEN")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# Cartelle da ignorare (non pertinenti per Sophie)
CARTELLE_ESCLUSE = {"report meta ads", "report meta", "arcanum"}

# Indice Drive: lista di {cartella, nome_file, file_id, mime}
DRIVE_INDEX = []
DRIVE_LOCK  = threading.Lock()

def get_google_access_token():
    """Ottiene un access token fresco usando il refresh token."""
    try:
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": GOOGLE_REFRESH_TOKEN,
            "grant_type":    "refresh_token"
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("access_token")
    except Exception as e:
        print(f"Errore token Google: {e}")
        return None

def lista_file_cartella(folder_id, access_token):
    """Ritorna la lista di file in una cartella Drive."""
    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id,name,mimeType)",
            "pageSize": "100"
        })
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("files", [])
    except Exception as e:
        print(f"Errore lista cartella {folder_id}: {e}")
        return []

def estrai_testo_pdf(contenuto_bytes):
    """Estrae testo da bytes PDF usando pymupdf."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=contenuto_bytes, filetype="pdf")
        testo = ""
        for page in doc:
            testo += page.get_text()
        doc.close()
        return testo.strip()
    except Exception as e:
        print(f"Errore estrazione PDF: {e}")
        return ""

def leggi_file_drive_bytes(file_id, access_token):
    """Scarica il contenuto binario di un file da Drive."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except Exception as e:
        print(f"Errore download file {file_id}: {e}")
        return None

def leggi_file_testo(file_id, mime, access_token):
    """Legge testo da .txt, Google Doc o PDF."""
    try:
        import urllib.request
        # Google Doc → export come testo
        if mime == "application/vnd.google-apps.document":
            req = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=text/plain",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        # PDF → scarica e estrai testo
        elif mime == "application/pdf":
            raw = leggi_file_drive_bytes(file_id, access_token)
            if raw:
                return estrai_testo_pdf(raw)
            return ""
        # .txt → scarica diretto
        else:
            raw = leggi_file_drive_bytes(file_id, access_token)
            if raw:
                return raw.decode("utf-8", errors="ignore")
            return ""
    except Exception as e:
        print(f"Errore lettura file {file_id}: {e}")
        return ""

def costruisci_indice_drive():
    """
    Scansiona tutte le sottocartelle di Arcanum/ e costruisce
    un indice {cartella, nome_file, file_id, mime}.
    Non scarica i file — solo registra dove sono.
    """
    global DRIVE_INDEX
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, GOOGLE_DRIVE_FOLDER_ID]):
        print("Google Drive: variabili mancanti, skip.")
        return

    access_token = get_google_access_token()
    if not access_token:
        print("Google Drive: impossibile ottenere access token.")
        return

    print("Drive: costruzione indice in corso...")
    nuovo_indice = []

    # Lista le sottocartelle di Arcanum/
    elementi = lista_file_cartella(GOOGLE_DRIVE_FOLDER_ID, access_token)
    for el in elementi:
        nome_cartella = el.get("name", "")
        mime          = el.get("mimeType", "")
        fid           = el.get("id", "")

        # Salta cartelle escluse
        if nome_cartella.lower() in CARTELLE_ESCLUSE:
            continue

        # Se è una cartella, scansiona il suo contenuto
        if mime == "application/vnd.google-apps.folder":
            file_dentro = lista_file_cartella(fid, access_token)
            for f in file_dentro:
                fn   = f.get("name", "")
                fm   = f.get("mimeType", "")
                ffid = f.get("id", "")
                # Accetta PDF, txt, Google Doc
                if fm in ("application/pdf", "text/plain",
                          "application/vnd.google-apps.document") or fn.endswith(".txt"):
                    nuovo_indice.append({
                        "cartella": nome_cartella,
                        "nome":     fn,
                        "id":       ffid,
                        "mime":     fm
                    })
        # File direttamente nella root di Arcanum (txt o Google Doc)
        elif mime in ("text/plain", "application/vnd.google-apps.document") or nome_cartella.endswith(".txt"):
            nuovo_indice.append({
                "cartella": "Generale",
                "nome":     nome_cartella,
                "id":       fid,
                "mime":     mime
            })

    with DRIVE_LOCK:
        DRIVE_INDEX = nuovo_indice

    print(f"Drive: indice costruito — {len(nuovo_indice)} file in {len(set(x['cartella'] for x in nuovo_indice))} cartelle.")

def cerca_in_drive(query, max_files=3):
    """
    Cerca i file Drive più pertinenti alla query,
    li scarica e restituisce il testo estratto.
    Esclude 'Report Meta Ads'.
    """
    with DRIVE_LOCK:
        indice = list(DRIVE_INDEX)

    if not indice:
        return "", False

    # Parole chiave dalla query
    parole = [w.lower() for w in re.sub(r'[^\w\s]', ' ', query).split() if len(w) > 2]

    # Calcola punteggio per ogni file in base a cartella + nome
    scored = []
    for entry in indice:
        cartella_norm = entry["cartella"].lower()
        nome_norm     = entry["nome"].lower()
        score = 0
        for p in parole:
            if p in cartella_norm: score += 5
            if p in nome_norm:     score += 3
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [e for _, e in scored[:max_files]]

    if not top:
        return "", False

    # Scarica e leggi il testo dei file selezionati
    access_token = get_google_access_token()
    if not access_token:
        return "", False

    testi = []
    for entry in top:
        testo = leggi_file_testo(entry["id"], entry["mime"], access_token)
        if testo.strip():
            # Prendi max 2000 caratteri per file per non sovraccaricare il prompt
            testi.append(f"[{entry['cartella']} / {entry['nome']}]\n{testo[:2000]}")
            print(f"Drive: letto {entry['nome']} ({len(testo)} caratteri)")

    if not testi:
        return "", False

    return "\n\n".join(testi), True

def aggiorna_drive_loop():
    """Thread in background: ricostruisce l'indice Drive ogni ora."""
    while True:
        costruisci_indice_drive()
        time.sleep(3600)



# ============================================================
# EMBEDDING E PRODOTTI
# ============================================================

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

# Percorso product_index.json compatibile con Railway
INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "product_index.json")

client    = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
app       = Flask(__name__)
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

# Avvia costruzione indice Drive e thread di aggiornamento
print("Costruzione indice Google Drive...")
costruisci_indice_drive()
t = threading.Thread(target=aggiorna_drive_loop, daemon=True)
t.start()

# Avvia loop email Sophie
t_email = threading.Thread(target=loop_email, daemon=True)
t_email.start()

# Avvia loop notifiche spedizione
t_notifiche = threading.Thread(target=loop_notifiche_spedizione, daemon=True)
t_notifiche.start()

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
    "impastatrice":   ["impastatrici","impasto","spirale"],
    "sfogliatrice":   ["sfogliatrici","sfoglia","laminatoio","stendi"],
    "sottovuoto":     ["confezionamento","conservazione","termosaldatura"],
    "lievitazione":   ["lievitare","lievito","fermalievitazione","cella","fermabiga"],
    "fermalievitazione": ["cella","lievitazione","fermabiga","armadio lievitazione"],
    "stagionatura":   ["maturazione","stagionare","celle stagionatura"],
    "abbattitore":    ["abbattimento","raffreddamento rapido","shock termico"],
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
        f"Seleziona SOLO i numeri dei prodotti STRETTAMENTE pertinenti alla richiesta del cliente. "
        f"Massimo {limit}. "
        f"REGOLE DI ESCLUSIONE RIGIDE:\n"
        f"- Escludi qualsiasi prodotto di categoria diversa da quella richiesta. "
        f"Esempio: se il cliente chiede teglie o stampi, escludi arrotondatrici, impastatrici, forni e qualsiasi macchinario. "
        f"Esempio: se il cliente chiede arrotondatrici, escludi teglie, stampi, carrelli. "
        f"Esempio: se il cliente chiede forni, escludi teglie e macchinari. "
        f"- Escludi prodotti che contengono casualmente una parola della query ma appartengono a una categoria diversa. "
        f"- Se nessun prodotto e davvero pertinente, rispondi con: nessuno\n"
        f"Rispondi SOLO con numeri separati da virgola (es: 1,3) oppure con la parola: nessuno"
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        risposta = msg.content[0].text.strip().lower()
        if "nessuno" in risposta:
            return []
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
.tg { background: #fff3cd; color: #856404; }
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

function buildBubble(text, sources, emails, products, drive) {
  var escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  escaped = escaped.replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank" style="color:#c0392b;font-weight:600;text-decoration:underline;">&#128279; Apri link</a>');
  var src = '<div class="src">';
  if (sources && sources.length) {
    for (var i = 0; i < sources.length; i++) src += '<span class="tag td">' + sources[i] + '</span>';
  }
  if (emails > 0) src += '<span class="tag te">' + emails + ' email</span>';
  if (drive) src += '<span class="tag tg">&#128194; Drive</span>';
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

function addMsg(text, role, sources, emails, products, drive) {
  var chat = document.getElementById('chat');
  var row = document.createElement('div');
  row.className = role === 'u' ? 'row user' : 'row';
  var av = document.createElement('div');
  av.className = 'mav ' + role;
  av.textContent = role === 's' ? 'S' : 'U';
  var bub = document.createElement('div');
  bub.className = 'bub ' + role;
  if (role === 's') {
    bub.innerHTML = buildBubble(text, sources, emails, products, drive);
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
  addMsg(text, 'u', [], 0, [], false);
  var loading = addMsg('...', 's', [], 0, [], false);
  loading.classList.add('loading');
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, cid: cid})
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    loading.classList.remove('loading');
    loading.innerHTML = buildBubble(data.response, data.doc_sources, data.email_count, data.products, data.drive_used);
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
        return jsonify({"response": "...", "doc_sources": [], "email_count": 0, "products": [], "drive_used": False})

    if cid not in histories:
        histories[cid] = []
    history = histories[cid]

    last = [m["content"] for m in history if m["role"] == "user"][-2:]
    q    = " ".join(last + [message])

    docs     = get_docs(q)
    emails   = get_emails(q)
    products = cerca_prodotti(q, limit=4)

    # ── TRACKING SPEDIZIONE ──────────────────────────────────
    tracking_ctx = ""
    TRACKING_KEYWORDS = [
        # Italiano — domande dirette sullo stato
        "spedito", "spedita", "spediti", "spedite",
        "è stato spedito", "è stata spedita", "sono stati spediti", "sono state spedite",
        "è partito", "è partita", "sono partiti", "sono partite",
        "è uscito", "è uscita", "sono usciti", "sono uscite",
        "ha spedito", "avete spedito", "l'avete mandato", "l'avete mandato",
        "quando spedite", "quando mandate", "quando parte",
        # Arrivo e consegna
        "quando arriva", "quando arriverà", "quando lo ricevo",
        "arrivato", "arrivata", "arrivati", "arrivate",
        "non è arrivato", "non è arrivata", "non sono arrivati", "non sono arrivate",
        "non ho ricevuto", "non ho ancora ricevuto", "aspetto ancora",
        "in ritardo", "ritardo", "doveva arrivare", "doveva venire",
        # Dove è
        "dov'è", "dove è", "dove sono", "dove si trova",
        "dov'è il mio", "dove sono i miei", "dove sono le mie",
        "dov'è la mia", "dov'è il pacco", "dov'è il corriere",
        # Termini generici spedizione
        "pacco", "pacchi", "paccone", "collo", "colli",
        "spedizione", "spedizioni", "consegna", "consegnato", "consegnata",
        "tracking", "tracciamento", "tracciare", "traccia",
        "codice tracking", "codice tracciamento", "codice spedizione",
        "numero spedizione", "numero tracking", "numero pacco",
        "numero corriere", "codice corriere", "link tracking",
        # Prodotti specifici Starpizza
        "teglie", "teglia", "carrelli", "carrello", "attrezzi", "attrezzo",
        "macchina", "macchinari", "arrotondatrice", "arrotondatrici",
        "impastatrice", "impastatrici", "sfogliatrice", "stampi", "stampo",
        "roba", "materiale", "materiali", "attrezzatura", "attrezzature",
        "acquisto", "acquisti", "ordine", "ordini", "cosa ho ordinato",
        # Corrieri
        "corriere", "brt", "gls", "dhl", "sda", "poste", "bartolini",
        "ups", "fedex", "nexive", "fercam", "tnt",
        # Stato spedizione
        "in transito", "fermo", "bloccato", "giacenza", "giacente",
        "tentativo di consegna", "assente", "non trovato",
        # Dialettale / informale italiano
        "l'avete mandato", "l'avete spedito", "me lo mandate",
        "me lo spedite", "quando me lo portano", "quando passa il corriere",
        "il corriere è passato", "è passato il corriere",
        # Inglese
        "track", "tracking", "shipment", "shipped", "delivery", "deliver",
        "where is", "where are", "has it shipped", "order status",
        "package", "parcel", "dispatch", "dispatched", "on its way",
        "out for delivery", "in transit",
        # Francese
        "suivi", "livraison", "expédié", "expédiée", "colis",
        "où est", "quand arrive", "numéro de suivi",
        # Spagnolo
        "seguimiento", "envío", "entrega", "dónde está", "dónde están",
        "pedido", "paquete", "despachado", "enviado",
        # Tedesco
        "sendung", "lieferung", "paket", "wo ist", "versandt",
        "tracking nummer", "lieferstatus", "wann kommt",
        # Polacco
        "śledzenie", "przesyłka", "dostawa", "gdzie jest", "kiedy dotrze",
        # Arabo (per mercato halal)
        "شحنة", "تتبع", "توصيل", "أين",
    ]
    msg_lower_track = message.lower()
    is_tracking_request = any(kw in msg_lower_track for kw in TRACKING_KEYWORDS)

    # Memoria contesto: se Sophie ha già chiesto info tracking nella sessione,
    # tratta anche la risposta successiva come richiesta tracking
    if not is_tracking_request and history:
        ultimi_sophie = [m["content"] for m in history if m["role"] == "assistant"][-2:]
        testo_sophie = " ".join(ultimi_sophie).lower()
        TRACKING_FOLLOWUP = ["numero d'ordine", "indirizzo email", "numero di telefono",
                             "order number", "email address", "phone number",
                             "numéro de commande", "número de pedido"]
        if any(kw in testo_sophie for kw in TRACKING_FOLLOWUP):
            is_tracking_request = True  # il cliente sta rispondendo alla domanda di Sophie

    if is_tracking_request:
        # Estrai possibile codice ordine / email / nome dal messaggio
        # Cerca prima nella sessione storica se il cliente ha già dato info
        history_testo = " ".join(
            m["content"] for m in history if m["role"] == "user"
        )
        search_query = message  # usa il messaggio corrente come query

        # Se il messaggio contiene email, usala come query principale
        testo_completo = message + " " + history_testo
        email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', testo_completo)
        if email_match:
            search_query = email_match.group(0)

        # Se il messaggio contiene numero di telefono, usalo come query
        tel_match = re.search(r'(?:\+?\d[\d\s\-]{7,14}\d)', testo_completo)
        if tel_match and not email_match:
            search_query = re.sub(r'[\s\-]', '', tel_match.group(0))

        tracking_info = spediamopro_tracking_testo(search_query)
        if tracking_info:
            tracking_ctx = (
                f"\n\n=== TRACKING SPEDIZIONE (dati in tempo reale) ===\n{tracking_info}\n"
                "Usa questi dati per rispondere al cliente in modo semplice e umano, come se parlassi con un artigiano. "
                "NON usare elenchi puntati. NON usare parole tecniche. "
                "Esempio: 'Il tuo pacco è in viaggio, arriva il 13 aprile 😊 Puoi seguirlo qui: [link]' "
                "Se è già consegnato dillo chiaramente. Se c'è un problema dillo con gentilezza e suggerisci di contattarci.\n"
            )
        else:
            tracking_ctx = (
                "\n\n=== TRACKING SPEDIZIONE ===\n"
                "Il cliente vuole sapere dove è la sua spedizione ma non ho ancora i suoi dati per cercarla. "
                "Chiedi SOLO questo, in modo semplice e amichevole, con una emoji: "
                "il numero di telefono o l'email con cui ha fatto l'ordine. "
                "NON usare liste puntate. NON usare parole tecniche come 'tracking' o 'numero d'ordine'. "
                "Esempio di risposta ideale: 'Certo! Dimmi il tuo numero di telefono o email e controllo subito 😊'\n"
            )
    # ── FINE TRACKING ────────────────────────────────────────

    # Cerca nei PDF Drive pertinenti alla domanda
    drive_ctx, drive_used = cerca_in_drive(q)

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
        rows_c = []
        msg_embedding = get_embedding(message) if OPENAI_API_KEY else None

        if msg_embedding:
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
                        if any(w.lower() in (dom or '').lower() for w in message.split() if len(w) > 2):
                            scored.append((0.5, dom, risp))

                scored.sort(reverse=True)
                rows_c = [(dom, risp) for score, dom, risp in scored[:3] if score > 0.3]

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

    # Istruzioni da Google Drive
    drive_section = ""
    if drive_ctx.strip():
        drive_section = f"\n\n=== ISTRUZIONI AGGIORNATE DAL TEAM (Google Drive) ===\n{drive_ctx[:3000]}\n"

    if docs:
        note = "Hai documentazione tecnica - usala con precisione."
    elif emails:
        note = f"Hai {len(emails)} email storiche simili - analizzale e riassumi."
    else:
        note = "Non hai info specifiche nel database. Dillo con naturalezza e suggerisci di contattare Starpizza."

    # Rileva lingua dal header Accept-Language (geolocalizzazione browser)
    lang_hint = ""
    detected_lang = "it"  # default italiano

    accept_lang = request.headers.get("Accept-Language", "").lower()
    msg_lower = message.lower()

    # Priorità 1: header browser (più affidabile)
    if accept_lang:
        if accept_lang.startswith("en"):
            detected_lang = "en"
        elif accept_lang.startswith("fr"):
            detected_lang = "fr"
        elif accept_lang.startswith("es"):
            detected_lang = "es"
        elif accept_lang.startswith("de"):
            detected_lang = "de"
        elif accept_lang.startswith("pl"):
            detected_lang = "pl"
        elif accept_lang.startswith("ar"):
            detected_lang = "ar"
        elif accept_lang.startswith("sr"):
            detected_lang = "sr"

    # Priorità 2: parole chiave nel messaggio (solo parole NON ambigue)
    # Italiano ha priorità assoluta — se ci sono parole italiane chiare, non override
    PAROLE_ITALIANE = ["avrei","vorrei","potrei","sarei","grazie","salve","ciao",
                       "buongiorno","buonasera","prego","gentilmente","cortesemente",
                       "fattura","ordine","spedizione","prodotto","prezzo","offerta",
                       "sono","siamo","abbiamo","voglio","vogliamo","devo","dobbiamo",
                       "questo","questa","questi","queste","mio","mia","nostro","nostra"]
    if any(w in msg_lower for w in PAROLE_ITALIANE):
        detected_lang = "it"  # italiano confermato, non fare override
    elif any(w in msg_lower for w in ["hello","hi ","good morning","good evening",
                                       "i need","i want","i would","please send",
                                       "thank you","thanks","could you","can you",
                                       "do you have","i am","we are","my name"]):
        detected_lang = "en"
    elif any(w in msg_lower for w in ["bonjour","merci","vous ","je ","est ","les ","des ","que ","pour "]):
        detected_lang = "fr"
    elif any(w in msg_lower for w in ["hola","gracias","tiene","como ","por favor","para ","esto "]):
        detected_lang = "es"
    elif any(w in msg_lower for w in ["danke","bitte","haben","ich ","sie ","die ","der ","das "]):
        detected_lang = "de"

    lang_map = {
        "en": "CRITICAL RULE: The customer is writing in ENGLISH. You MUST reply in ENGLISH only. Never use Italian.\n\n",
        "fr": "REGLE CRITIQUE: Le client ecrit en francais. Tu DOIS repondre en FRANCAIS uniquement.\n\n",
        "es": "REGLA CRITICA: El cliente escribe en espanol. DEBES responder en ESPANOL unicamente.\n\n",
        "de": "KRITISCHE REGEL: Der Kunde schreibt auf Deutsch. Du MUSST auf DEUTSCH antworten.\n\n",
        "pl": "KRYTYCZNA ZASADA: Klient pisze po polsku. MUSISZ odpowiadac po POLSKU.\n\n",
        "ar": "قاعدة حاسمة: العميل يكتب بالعربية. يجب أن ترد بالعربية فقط.\n\n",
        "sr": "KRITIČNO PRAVILO: Klijent piše na srpskom. MORATE odgovoriti na SRPSKOM.\n\n",
    }
    lang_hint = lang_map.get(detected_lang, "")

    # Traduzione automatica correzioni nella lingua del cliente
    auto_translate_note = ""
    if detected_lang != "it" and correzioni_ctx:
        auto_translate_note = f"\nIMPORTANTE: Le correzioni del team qui sopra sono in italiano. Traducile automaticamente in {detected_lang.upper()} prima di usarle nella risposta. Non rispondere mai in italiano se il cliente usa un'altra lingua.\n"

    system = (
        lang_hint +
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
        "- DATI DI CONTATTO: NON inventare MAI numeri di telefono, indirizzi email, URL, indirizzi fisici o qualsiasi dato di contatto che non sia esplicitamente presente nelle istruzioni o nella documentazione fornita. Se un cliente chiede un contatto che non hai, rispondi ESATTAMENTE: 'Per questa informazione ti invito a visitare starpizza.org o a scriverci tramite il sito.' Non improvvisare mai.\n"
        + auto_translate_note
        + drive_section + tracking_ctx + docs_ctx + email_ctx + products_ctx + correzioni_ctx
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
        salva_chat(cid, "cliente", message)
        salva_chat(cid, "sophie", response)
        return jsonify({
            "response":    response,
            "doc_sources": doc_sources,
            "email_count": len(emails),
            "products":    products,
            "drive_used":  drive_used
        })
    except Exception as e:
        history.pop()
        return jsonify({"response": f"Errore: {str(e)}", "doc_sources": [], "email_count": 0, "products": [], "drive_used": False})


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

    # Stato Drive
    with DRIVE_LOCK:
        drive_count = len(DRIVE_INDEX)

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
.stat.drive h3 { color: #856404; }
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
<div class="stat drive"><h3>""" + str(drive_count) + """</h3><p>&#128194; File Drive indicizzati</p></div>
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


@app.route("/admin/drive")
def admin_drive():
    """Pagina di debug per vedere i file Drive indicizzati."""
    with DRIVE_LOCK:
        indice = list(DRIVE_INDEX)

    cartelle = {}
    for entry in indice:
        c = entry["cartella"]
        if c not in cartelle:
            cartelle[c] = []
        cartelle[c].append(entry["nome"])

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Drive Index</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 24px; }
h1 { color: #c0392b; margin-bottom: 20px; }
.cartella { background: white; border-radius: 10px; padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.07); }
.cartella h3 { color: #856404; margin-bottom: 10px; font-size: 1rem; }
.file { font-size: 0.9rem; color: #444; padding: 4px 0; border-bottom: 1px solid #f0f0f0; }
.file:last-child { border: none; }
.stats { background: #c0392b; color: white; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; }
a { color: #c0392b; }
</style></head><body>
<h1>📁 File Drive Indicizzati</h1>"""

    html += f'<div class="stats">Totale: <strong>{len(indice)} file</strong> in <strong>{len(cartelle)} cartelle</strong> — <a href="/admin/drive" style="color:white;">🔄 Aggiorna</a> | <a href="/admin/chat" style="color:white;">← Pannello</a></div>'

    if not cartelle:
        html += '<p>Nessun file trovato. Controlla le variabili Google su Railway.</p>'
    else:
        for nome_cartella, files in sorted(cartelle.items()):
            html += f'<div class="cartella"><h3>📂 {nome_cartella} ({len(files)} file)</h3>'
            for f in files:
                html += f'<div class="file">📄 {f}</div>'
            html += '</div>'

    html += '</body></html>'
    return html
