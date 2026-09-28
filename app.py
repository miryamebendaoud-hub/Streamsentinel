"""StreamSentinel — demo interface.

Three spaces:
- New observation: the citizen sends a photo, the Colab API (GPU) analyses it live,
  then the citizen checks and completes the pre-filled form before sending it;
- Analysed examples: the photos of the test set, analysed in advance (works without Colab);
- Manager: validation queue for high-risk observations.

Run from the project root:  python -m streamlit run app.py
"""
import base64
import json
import math
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

from fhir_export import record_to_fhir
from i18n import tr, level, SOURCES, ORIENTATIONS

DEMO = Path(__file__).parent / "demo"
DECISIONS = DEMO / "decisions.json"
OBSERVATIONS = DEMO / "observations"
SITES = DEMO / "sites_oah.json"   # OneAquaHealth research sites (ENORA API)

st.set_page_config(page_title="StreamSentinel", page_icon="🌊", layout="wide")

# ---------- OneAquaHealth form (from the J4 notebook) ----------
# field: (step of the citizen app, options)
OPTIONS = {
    "channel_form": (4, ["FLAT", "U_SHAPE", "V_SHAPE"]),
    "bottom_type": (4, ["NAT", "ARTIFICIAL"]),
    "bank_type": (4, ["NAT", "ARTIFICIAL", "LAYED_STONES"]),
    "water_flow": (4, ["FAS", "SLOW", "STAGNANT", "DRY"]),
    "water_aspect": (5, ["CLEAR", "MUDDY", "FOAM", "ALTERED_COLOR"]),
    "barriers": (5, ["YES", "NO"]),
    "draining_pipes": (5, ["YES", "NO"]),
    "sewage_discharge": (5, ["YES", "NO"]),
    "construction": (5, ["YES", "NO"]),
    "impervious_left": (6, ["YES", "NO"]),
    "impervious_right": (6, ["YES", "NO"]),
    "vegetation_left": (6, ["YES", "NO"]),
    "vegetation_right": (6, ["YES", "NO"]),
    "vegetation_type_left": (6, ["HERBS", "SHRUBS", "TREES"]),
    "vegetation_type_right": (6, ["HERBS", "SHRUBS", "TREES"]),
    "vegetation_cuts": (6, ["YES", "NO"]),
}
ETAPES = {4: "Overview over 100 m", 5: "Water and structures", 6: "Banks over 5–10 m"}

NOMS_CHAMPS = {
    "channel_form": "Channel shape",
    "bottom_type": "Bottom type",
    "bank_type": "Bank type",
    "water_flow": "Water flow",
    "water_aspect": "Water appearance",
    "barriers": "Barriers (weirs, dams)",
    "draining_pipes": "Drainage pipes",
    "sewage_discharge": "Sewage discharge",
    "construction": "Construction work",
    "impervious_left": "Impervious surface, left bank",
    "impervious_right": "Impervious surface, right bank",
    "vegetation_left": "Vegetation, left bank",
    "vegetation_right": "Vegetation, right bank",
    "vegetation_type_left": "Vegetation type, left bank",
    "vegetation_type_right": "Vegetation type, right bank",
    "vegetation_cuts": "Vegetation cutting",
    "research_site": "Research site",
    "water_height": "Water height",
    "water_withdrawal": "Water withdrawal",
    "habitats": "Habitats",
    "natural_debris": "Natural debris",
    "invasive_species": "Invasive species",
    "feelings": "Feelings",
}
VALEURS = {
    "FLAT": "Flat", "U_SHAPE": "U-shaped", "V_SHAPE": "V-shaped",
    "NAT": "Natural", "ARTIFICIAL": "Artificial", "LAYED_STONES": "Laid stones",
    "FAS": "Fast", "SLOW": "Slow", "STAGNANT": "Stagnant", "DRY": "Dry",
    "CLEAR": "Clear", "MUDDY": "Muddy", "FOAM": "Foam", "ALTERED_COLOR": "Altered colour",
    "YES": "Yes", "NO": "No",
    "HERBS": "Herbs", "SHRUBS": "Shrubs", "TREES": "Trees",
    "GOOD": "Good", "MODERATE": "Moderate", "POOR": "Poor",
}
COULEURS = {"FAIBLE": "green", "MODERE": "orange", "ELEVE": "red"}


# ---------- Outils ----------
def nom(chemin):
    return Path(str(chemin)).name


def norm(niveau):
    """ÉLEVÉ, ELEVE, élevé -> ELEVE"""
    return unicodedata.normalize("NFKD", str(niveau)).encode("ascii", "ignore").decode().upper()


def nom_champ(cle):
    return NOMS_CHAMPS.get(cle, cle.replace("_", " ").capitalize())


def libelle(code):
    if code in (None, "NOT_SURE"):
        return "To complete"
    return VALEURS.get(code, str(code))


def en_liste(data):
    return data if isinstance(data, list) else list(data.values())


def url_fixe():
    """Permanent API address (ngrok domain), read from .streamlit/secrets.toml."""
    try:
        return str(st.secrets.get("API_URL", "")).strip().rstrip("/")
    except Exception:   # no secrets.toml file
        return ""


def url_api():
    return url_fixe() or st.session_state.get("url_api", "").strip().rstrip("/")


# free ngrok shows a warning page to browsers: this header skips it
ENTETES = {"ngrok-skip-browser-warning": "true"}


@st.cache_data
def charger():
    j4 = json.loads((DEMO / "resultats_j4.json").read_text(encoding="utf-8"))
    j5 = json.loads((DEMO / "resultats_j5.json").read_text(encoding="utf-8"))
    fiches = en_liste(j4)
    risques = {nom(r["photo"]): r for r in en_liste(j5)}
    return fiches, risques


def chemin_image(fiche):
    if not fiche:
        return None
    brut = fiche.get("image_annotee") or ""
    if brut and Path(brut).exists():
        return Path(brut)
    p = DEMO / "annotees" / nom(brut or fiche.get("photo", ""))
    return p if p.exists() else None


def lire_json(chemin, defaut):
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else defaut


def ecrire_decision(cle, decision):
    d = lire_json(DECISIONS, {})
    d[cle] = {"decision": decision, "date": datetime.now().isoformat(timespec="minutes")}
    DECISIONS.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def lire_observations():
    if not OBSERVATIONS.exists():
        return []
    obs = [json.loads(p.read_text(encoding="utf-8")) for p in OBSERVATIONS.glob("*.json")]
    return sorted(obs, key=lambda o: o["date"], reverse=True)


def enregistrer_observation(analyse, reponses, evaluation, corrections):
    OBSERVATIONS.mkdir(parents=True, exist_ok=True)
    ident = analyse["id"]
    image = OBSERVATIONS / f"{ident}.jpg"
    image.write_bytes(base64.b64decode(analyse["image_annotee_b64"]))
    fiche = dict(analyse["fiche"], image_annotee=str(image))
    obs = {
        "id": ident,
        "date": datetime.now().isoformat(timespec="minutes"),
        "fiche": fiche,
        "risque": analyse["risque"],
        "reponses_citoyen": reponses,
        "evaluation_citoyen": evaluation,
        "corrections": corrections,
        "lieu": analyse.get("lieu"),
    }
    (OBSERVATIONS / f"{ident}.json").write_text(json.dumps(obs, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- OneAquaHealth research sites ----------
def _premier(d, cles):
    for c in cles:
        v = d.get(c)
        if v not in (None, ""):
            return v.get("name", v.get("nom")) if isinstance(v, dict) else v
    return None


@st.cache_data
def charger_sites():
    """Read demo/sites_oah.json (raw answer of api.enora-oah.eu/api/sites/all), whatever its exact format."""
    if not SITES.exists():
        return []
    brut = json.loads(SITES.read_text(encoding="utf-8"))
    if isinstance(brut, dict):
        brut = next((brut[k] for k in ("data", "sites", "results", "items") if isinstance(brut.get(k), list)),
                    list(brut.values()))
    sites = []
    for x in brut:
        if not isinstance(x, dict):
            continue
        lat, lon = _premier(x, ["latitude", "lat", "y"]), _premier(x, ["longitude", "lon", "lng", "long", "x"])
        for cle in ("location", "position", "coordinates", "geometry"):
            if lat is None and isinstance(x.get(cle), dict):
                sous = x[cle]
                lat, lon = _premier(sous, ["latitude", "lat", "y"]), _premier(sous, ["longitude", "lon", "lng", "x"])
                if lat is None and isinstance(sous.get("coordinates"), list):
                    lon, lat = sous["coordinates"][:2]
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        sites.append({
            "code": str(_premier(x, ["code", "site_code", "siteCode", "reference", "id"]) or ""),
            "nom": str(_premier(x, ["name", "nom", "site_name", "siteName", "label", "title"]) or ""),
            "ville": str(_premier(x, ["city", "ville", "town", "municipality", "case_study", "caseStudy"]) or ""),
            "lat": lat, "lon": lon,
        })
    return sites


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a))


def site_proche(sites, lat, lon):
    if not sites:
        return None, None
    s = min(sites, key=lambda x: distance_km(lat, lon, x["lat"], x["lon"]))
    return s, distance_km(lat, lon, s["lat"], s["lon"])


def nom_site(site):
    return " · ".join(v for v in (site.get("ville"), site.get("code"), site.get("nom")) if v)


def choisir_lieu():
    """Map: click an official site (blue dots) or anywhere on the map."""
    sites = charger_sites()
    lieu = st.session_state.get("lieu")

    st.markdown("**Where are you?**")
    if sites:
        st.caption(f"Click one of the {len(sites)} OneAquaHealth research sites (blue dots), "
                   "or anywhere on the map if you are elsewhere.")
    else:
        st.caption("Click the map where the photo was taken. (OneAquaHealth site list missing: "
                   "add demo/sites_oah.json.)")

    if lieu:
        centre, zoom = [lieu["lat"], lieu["lon"]], 13
    elif sites:
        centre = [sum(x["lat"] for x in sites) / len(sites), sum(x["lon"] for x in sites) / len(sites)]
        zoom = 4
    else:
        centre, zoom = [46.6, 2.4], 5
    carte = folium.Map(location=centre, zoom_start=zoom, tiles="OpenStreetMap")
    for x in sites:
        folium.CircleMarker([x["lat"], x["lon"]], radius=6, color="#1d4ed8", fill=True,
                            fill_opacity=0.8, tooltip=nom_site(x)).add_to(carte)
    if lieu:
        folium.Marker([lieu["lat"], lieu["lon"]], icon=folium.Icon(color="red", icon="camera", prefix="fa"),
                      tooltip="Your observation").add_to(carte)
    retour = st_folium(carte, height=420, use_container_width=True, key="carte",
                       returned_objects=["last_clicked", "last_object_clicked"]) or {}

    clic = None
    for cle in ("last_object_clicked", "last_clicked"):
        v = retour.get(cle)
        if v and v != st.session_state.get(f"_{cle}"):
            st.session_state[f"_{cle}"] = v
            clic = clic or v
    if clic:
        lat, lon = clic["lat"], clic["lng"]
        site, dist = site_proche(sites, lat, lon)
        if site and dist <= 0.05:      # click on a blue dot: snap to the exact site
            lat, lon = site["lat"], site["lon"]
        st.session_state["lieu"] = {"lat": lat, "lon": lon,
                                    "site": site if site and dist <= 1 else None,
                                    "distance_m": round(dist * 1000) if site else None}
        st.rerun()

    if lieu:
        if lieu.get("site"):
            st.success(f"Research site: **{nom_site(lieu['site'])}**"
                       + (f" ({lieu['distance_m']} m away)" if lieu.get("distance_m", 0) > 50 else ""))
        else:
            st.info(f"Selected point: {lieu['lat']:.4f}, {lieu['lon']:.4f}. "
                    "No OneAquaHealth site within 1 km: observation outside the research network.")
        st.caption("River flow data (Hub'Eau) is only available in France, e.g. in Toulouse.")
    return st.session_state.get("lieu")


def bouton_fhir(cle, fiche, risque, obs=None, decision=None):
    """Download the FHIR R4 Bundle (status "final" only once a manager has validated it)."""
    valide = bool(decision and decision.get("decision") in ("alert validated", "alerte validée"))
    bundle = record_to_fhir(fiche, risque,
                            answers=(obs or {}).get("reponses_citoyen"),
                            rating=(obs or {}).get("evaluation_citoyen"),
                            place=(obs or {}).get("lieu"), validated=valide)
    st.download_button(
        "Export as FHIR", json.dumps(bundle, indent=2, ensure_ascii=False),
        file_name=f"streamsentinel_{Path(str(cle)).stem}.fhir.json", mime="application/fhir+json",
        key=f"fhir_{cle}",
        help="FHIR R4 Bundle aligned with the OneAquaHealth implementation guide (hl7-eu/oah). "
             "Status 'final' once the alert is validated, 'preliminary' otherwise.")


# ---------- Shared display blocks ----------
def ligne_niveau(titre, bloc, avec_score=True):
    couleur = COULEURS.get(norm(bloc["niveau"]), "gray")
    score = f" (score {bloc['score']})" if avec_score else ""
    st.markdown(f"**{titre}**: :{couleur}[**{level(bloc['niveau'])}**]{score}")
    st.caption(tr(bloc.get("consigne", "")))


def afficher_risque(r, detaille=True):
    st.subheader("Risk level")
    ligne_niveau("Humans", r["risque_humains"], detaille)
    ligne_niveau("Animals", r["risque_animaux"], detaille)
    st.caption("Indicative estimate, computed automatically: it does not replace "
               "official health advice.")
    if not detaille:
        if r.get("a_valider_gestionnaire"):
            st.warning("A manager will check your observation. No alert is issued without them.")
        return
    afficher_contexte(r)


def afficher_contexte(r, dans_encadre=False):
    # Streamlit forbids an expander inside another one: write the reasons directly
    bloc = st.container() if dans_encadre else st.expander("Why this level?")
    with bloc:
        if r.get("regles"):
            # rule inspector: each point of the score, with the evidence behind the rule
            st.dataframe(pd.DataFrame([{
                "Rule triggered": g["label"],
                "Humans": f"+{g['humains']}", "Animals": f"+{g['animaux']}",
                "Evidence": "; ".join(g["sources"]),
            } for g in r["regles"]]), hide_index=True, use_container_width=True,
                column_config={"Evidence": st.column_config.TextColumn("Evidence", width="large")})
            st.caption("Levels: 0–1 low, 2–3 moderate, 4 or more high. The OneAquaHealth factsheets "
                       "(Schmeller et al., 2026, doi:10.5281/zenodo.20345207) explain why a signal matters "
                       "for health; they give no thresholds for photo observations, so points and "
                       "thresholds are the team's choices.")
        else:
            for raison in r.get("raisons", []):
                st.write("•", tr(raison))
        if r.get("tendance_3j"):
            st.write("• 3-day trend:", tr(r["tendance_3j"]))

    m, h = r.get("meteo", {}), r.get("hubeau", {})
    c1, c2, c3 = st.columns(3)
    ratio = h.get("ratio")
    c1.metric("Rain, last 72 h", f"{m.get('pluie_72h_mm', '–')} mm", help="Rain over the last 3 days")
    c2.metric("Rain, next 3 days", f"{m.get('pluie_prevue_3j_mm', '–')} mm", help="Forecast rain over the next 3 days")
    c3.metric("River flow", f"{ratio * 100:.0f} %" if ratio is not None else "–",
              help="Current flow as a % of the 30-day median flow")
    if h.get("station"):
        st.caption(f"Flow measured at: {h['station']} ({h.get('distance_km')} km)")
    elif h and not h.get("disponible", True):
        st.caption(f"Flow unavailable: {tr(h.get('raison', 'no gauging station nearby'))}")

    if r.get("a_valider_gestionnaire"):
        st.warning("Sent to the manager. No alert is issued without their validation.")


def afficher_signalements(fiche):
    s = fiche.get("signalements", {})
    am = s.get("animal_mort", {})
    if am.get("valeur") == "YES":
        st.error(f"Possible dead animal: {tr(am.get('raison', ''))}. To be confirmed by a human.")
    dechets = s.get("dechets") or []
    if dechets:
        st.warning(f"{len(dechets)} litter item(s) detected in the photo (yellow boxes).")
    if am.get("valeur") != "YES" and not dechets:
        st.info("No litter or dead animal spotted by the AI. It can miss some: "
                "please check on site.")


def afficher_formulaire(fiche):
    """Read-only table (analysed examples)."""
    lignes = []
    for cle, rep in fiche.get("formulaire", {}).items():
        if not isinstance(rep, dict):
            continue
        valeur = rep.get("valeur")
        lignes.append({
            "Question": nom_champ(cle),
            "Proposed answer": libelle(valeur),
            "Confidence": int(round((rep.get("confiance") or 0) * 100)),
            "Source": SOURCES.get(rep.get("source", ""), rep.get("source", "")),
            "Justification": ("The model is not sure enough: to be checked on site"
                              if valeur == "NOT_SURE" else tr(rep.get("raison", ""))),
        })
    st.dataframe(
        pd.DataFrame(lignes),
        hide_index=True,
        height=38 + 35 * len(lignes),
        column_config={
            "Question": st.column_config.TextColumn("Question", width="medium"),
            "Proposed answer": st.column_config.TextColumn("Proposed answer", width="medium"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100,
                                                         format="%d%%", width="medium"),
            "Source": st.column_config.TextColumn("Source", width="small"),
            "Justification": st.column_config.TextColumn("Justification", width="large"),
        },
    )


def afficher_questions_laissees(fiche):
    laisses = fiche.get("laisses_au_citoyen", {})
    if laisses:
        with st.expander(f"{len(laisses)} questions are left for you to answer in the OneAquaHealth app"):
            for cle, pourquoi in laisses.items():
                st.write(f"• **{nom_champ(cle)}**: {tr(pourquoi)}")


# ---------- "New observation" space ----------
def lancer_analyse(url, photo, orientation, lieu):
    lat, lon = lieu["lat"], lieu["lon"]
    try:
        rep = requests.post(f"{url}/analyser",
                            files={"photo": (photo.name, photo.getvalue(), photo.type)},
                            data={"orientation": orientation, "lat": lat, "lon": lon},
                            headers=ENTETES, timeout=60)
        rep.raise_for_status()
        ident = rep.json()["id"]
    except requests.RequestException as e:
        st.error("The analysis service is not responding: the Colab notebook must be running. "
                 f"({e})")
        return

    with st.status("Sending the photo", expanded=True) as statut:
        derniere, debut = None, time.time()
        while time.time() - debut < 600:
            time.sleep(3)
            try:
                t = requests.get(f"{url}/resultat/{ident}", headers=ENTETES, timeout=30).json()
            except (requests.RequestException, ValueError):
                continue
            etat = t.get("etat")
            if etat is None:
                statut.update(label="Analysis not found", state="error")
                st.error(t.get("detail", "Unexpected answer from the API."))
                return
            if tr(t.get("etape")) != derniere:
                derniere = tr(t.get("etape"))
                st.write(derniere)
            statut.update(label=f"{derniere} ({t.get('ecoule_s', 0)} s)")
            if etat == "termine":
                statut.update(label=f"Analysis complete in {t.get('duree_s')} s", state="complete")
                st.session_state["analyse"] = dict(t["resultat"], id=ident, lieu=lieu)
                st.rerun()
            if etat == "erreur":
                statut.update(label="The analysis failed", state="error")
                st.error(t.get("erreur", "Unknown error on the Colab side."))
                return
        statut.update(label="The analysis is taking more than 10 minutes", state="error")


def ecran_envoi():
    url = url_api()
    if not url:
        st.info("To analyse a photo, paste the API address in the left menu, under "
                "'Analysis connection'. Meanwhile, open 'Analysed examples'.")
    else:
        try:
            requests.get(f"{url}/sante", headers=ENTETES, timeout=8).raise_for_status()
        except requests.RequestException:
            st.warning("The live analysis service is currently offline. "
                       "You can browse the 'Analysed examples' in the meantime.")

    photo = st.file_uploader("Photo of the stream", type=["jpg", "jpeg", "png"],
                             help="Frame the water and both banks, with no recognisable person.")
    orientation = st.radio(
        "Which way were you facing?", ["aval", "amont"], horizontal=True,
        format_func=lambda o: "Downstream (the current flows away from you)" if o == "aval"
        else "Upstream (facing the current)",
        help="Left and right banks are defined when looking downstream.")
    lieu = choisir_lieu()
    if photo:
        st.image(photo, width=420)

    if not lieu:
        st.caption("Pick the location on the map to start the analysis.")
    if st.button("Analyse the photo", type="primary", disabled=not (photo and url and lieu)):
        lancer_analyse(url, photo, orientation, lieu)


# ---------- Guided citizen form (same steps as the OneAquaHealth app) ----------
# Questions whose AI answer is hidden: on the independent evaluation (21 Wikimedia photos),
# their accuracy was not distinguishable from chance. The citizen answers them alone.
NO_PREFILL = {"channel_form", "vegetation_left", "vegetation_right", "overall"}

HELP = {
    "channel_form": ("What is the shape of the stream bed over about 100 m?",
                     {"FLAT": "wide and shallow bed", "U_SHAPE": "rounded bed, gentle banks",
                      "V_SHAPE": "narrow, deep bed with steep banks"}),
    "bottom_type": ("What is the bottom of the stream made of?",
                    {"NAT": "gravel, sand, mud or rock", "ARTIFICIAL": "concrete, paving or tiles"}),
    "bank_type": ("What are the banks made of?",
                  {"NAT": "soil and vegetation", "ARTIFICIAL": "concrete or walls",
                   "LAYED_STONES": "stones placed without concrete"}),
    "water_flow": ("How is the water moving?",
                   {"FAS": "waves or high velocity", "SLOW": "moving gently",
                    "STAGNANT": "standing water or isolated pools", "DRY": "no water in the bed"}),
    "water_aspect": ("What does the water look like? Choose the most striking sign.",
                     {"CLEAR": "you can see the bottom", "MUDDY": "brown or cloudy",
                      "FOAM": "white or greyish foam", "ALTERED_COLOR": "unusual colour, film or sheen"}),
    "water_withdrawal": ("Is water being pumped or taken out of the stream?", {}),
    "barriers": ("Is there a weir, dam or sluice blocking the flow?", {}),
    "draining_pipes": ("Do pipes flow into the stream?", {}),
    "sewage_discharge": ("Is there a wastewater outlet, grey water or a sewage smell?", {}),
    "construction": ("Is there construction work or machinery nearby?", {}),
    "impervious_left": ("Is more than a third of the left margin covered by roads, buildings or concrete?", {}),
    "impervious_right": ("Is more than a third of the right margin covered by roads, buildings or concrete?", {}),
    "vegetation_left": ("Is there vegetation on the first 5 metres of the left bank?", {}),
    "vegetation_right": ("Is there vegetation on the first 5 metres of the right bank?", {}),
    "vegetation_type_left": ("Which vegetation covers more than half of the first 5 metres?",
                             {"HERBS": "under 1.5 m", "SHRUBS": "1.5 to 3 m", "TREES": "over 3 m"}),
    "vegetation_type_right": ("Which vegetation covers more than half of the first 5 metres?",
                              {"HERBS": "under 1.5 m", "SHRUBS": "1.5 to 3 m", "TREES": "over 3 m"}),
    "vegetation_cuts": ("Has vegetation recently been cut on one or both banks?", {}),
}
YES_NO = ["YES", "NO"]
HABITATS = {"SB": "Sand banks", "SI": "Sand islands", "SD": "Stone deposits",
            "RRF": "Riffles, rapids, falls", "AV": "Aquatic vegetation"}
DEBRIS = {"FT": "Fallen trees", "FB": "Fallen branches", "FL": "Deposits of fallen leaves"}
CONDITIONAL = {"vegetation_type_left": "vegetation_left", "vegetation_type_right": "vegetation_right"}

STEPS = [
    ("Overview over 100 m", ["channel_form", "bottom_type", "bank_type", "#habitats", "#natural_debris",
                             "water_flow"]),
    ("Water and structures", ["water_aspect", "water_withdrawal", "barriers", "draining_pipes",
                              "sewage_discharge", "construction", "#water_height"]),
    ("Banks over 5 to 10 m", ["impervious_left", "impervious_right", "vegetation_left",
                              "vegetation_type_left", "vegetation_right", "vegetation_type_right",
                              "#invasive_species", "vegetation_cuts"]),
    ("Overall assessment", ["#overall"]),
    ("Your feelings", ["#feelings"]),
    ("Summary", ["#summary"]),
]
OPTIONS_ALL = {**{k: v[1] for k, v in OPTIONS.items()}, "water_withdrawal": YES_NO}

st.markdown("""<style>
.ss-tag {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.82rem;
         margin:2px 0 6px 0; line-height:1.5}
.ss-ai {background:#DDF0EC; color:#0B5345}
.ss-you {background:#FBEBD0; color:#7A4A06}
.ss-todo {background:#E8EAF0; color:#353B4A}
.ss-off {background:#EEE6F5; color:#4B2D66}
</style>""", unsafe_allow_html=True)


def _tag(css, text):
    st.markdown(f'<span class="ss-tag {css}">{text}</span>', unsafe_allow_html=True)


def _ai(formulaire, cle):
    """AI answer to show, or None (not sure, or hidden after evaluation)."""
    if cle in NO_PREFILL:
        return None
    v = formulaire.get(cle, {}).get("valeur")
    return v if v in OPTIONS_ALL.get(cle, []) else None


def _state(a):
    """Citizen answers, kept across steps (widgets are rebuilt at each step)."""
    k = f"form_{a['id']}"
    if k not in st.session_state:
        formulaire = a["fiche"].get("formulaire", {})
        st.session_state[k] = {"step": 0, "answers": {c: _ai(formulaire, c) for c in OPTIONS_ALL}}
    return st.session_state[k]


def _question(a, cle, answers):
    formulaire = a["fiche"].get("formulaire", {})
    parent = CONDITIONAL.get(cle)
    if parent and answers.get(parent) != "YES":
        answers[cle] = None
        return
    question, details = HELP.get(cle, (nom_champ(cle), {}))
    options = OPTIONS_ALL[cle] + ["NOT_SURE"]
    with st.container(border=True):
        st.markdown(f"**{nom_champ(cle)}**")
        current = answers.get(cle)
        choice = st.radio(question, options, horizontal=True, key=f"w_{a['id']}_{cle}",
                          index=options.index(current) if current in options else None,
                          format_func=lambda o: "I'm not sure" if o == "NOT_SURE" else libelle(o))
        answers[cle] = choice
        if details:
            st.caption("  \n".join(f"{libelle(o)}: {d}" for o, d in details.items()))
        ai = _ai(formulaire, cle)
        rep = formulaire.get(cle, {})
        if cle in NO_PREFILL:
            _tag("ss-off", "For you to answer: the AI is not reliable enough on this question yet")
        elif cle not in formulaire:
            _tag("ss-todo", "For you to answer: not pre-filled by the AI yet")
        elif ai is None:
            _tag("ss-todo", "For you to answer: the AI could not tell from the photo")
        elif choice == ai:
            conf = int(round((rep.get("confiance") or 0) * 100))
            src = SOURCES.get(rep.get("source", ""), rep.get("source", "AI"))
            _tag("ss-ai", f"Suggested by the AI, {conf}% confident ({src})")
            if rep.get("raison"):
                st.caption(f"Why: {tr(rep['raison'])}")
        else:
            _tag("ss-you", f"Changed by you. The AI suggested: {libelle(ai)}")


def _checklist(a, cle, question, choices, answers):
    with st.container(border=True):
        st.markdown(f"**{nom_champ(cle)}**")
        present = st.checkbox(question, value=bool(answers.get(cle)), key=f"w_{a['id']}_{cle}_any")
        picked = []
        if present:
            cols = st.columns(2)
            for i, (code, text) in enumerate(choices.items()):
                if cols[i % 2].checkbox(text, value=code in (answers.get(cle) or []),
                                        key=f"w_{a['id']}_{cle}_{code}"):
                    picked.append(code)
        answers[cle] = picked
        _tag("ss-todo", "For you to answer: not pre-filled by the AI yet")


def _water_height(a, answers):
    with st.container(border=True):
        st.markdown("**Water height**")
        answers["water_height"] = st.number_input(
            "What is the water height? Measure it on site as the OneAquaHealth protocol describes.",
            value=answers.get("water_height"), step=1.0, key=f"w_{a['id']}_water_height")
        _tag("ss-todo", "For you to answer: a physical measurement cannot be read from a photo")


def _invasive(a, answers):
    with st.container(border=True):
        st.markdown("**Invasive species**")
        options = ["YES", "NO", "NOT_SURE"]
        cur = answers.get("invasive_species")
        v = st.radio("Do you see any non-native or invasive plant species?", options, horizontal=True,
                     index=options.index(cur) if cur in options else None, key=f"w_{a['id']}_invasive",
                     format_func=lambda o: "I'm not sure" if o == "NOT_SURE" else libelle(o))
        answers["invasive_species"] = v
        if v == "YES":
            answers["invasive_which"] = st.text_input("Which ones?", value=answers.get("invasive_which", ""),
                                                      key=f"w_{a['id']}_invasive_which")
        _tag("ss-todo", "For you to answer: recognising a species needs a trained eye")


def _overall(a, answers):
    notes = ["GOOD", "MODERATE", "POOR"]
    with st.container(border=True):
        st.markdown("**Overall condition of the stream**")
        cur = answers.get("overall")
        answers["overall"] = st.radio(
            "Based on everything you saw, how would you rate this stretch of stream?", notes,
            horizontal=True, index=notes.index(cur) if cur in notes else None,
            format_func=libelle, key=f"w_{a['id']}_overall")
        _tag("ss-off", "For you to answer: the AI is not reliable enough on this question yet")


def _feelings(a, answers):
    with st.container(border=True):
        st.markdown("**How does this place make you feel?**")
        st.caption("Your personal experience of the place. The AI never fills this in.")
        feelings = answers.get("feelings") or {}
        na = st.checkbox("Not applicable", value=feelings == "NA", key=f"w_{a['id']}_feel_na")
        if na:
            answers["feelings"] = "NA"
            return
        values = feelings if isinstance(feelings, dict) else {}
        answers["feelings"] = {f: st.slider(f.capitalize(), 1, 5, values.get(f, 3), key=f"w_{a['id']}_feel_{f}")
                               for f in ("joy", "serenity", "anger", "fear")}


def _summary(a, answers):
    formulaire = a["fiche"].get("formulaire", {})
    rows = []
    for title, keys in STEPS[:3]:
        for cle in keys:
            if cle.startswith("#") or (CONDITIONAL.get(cle) and answers.get(CONDITIONAL[cle]) != "YES"):
                continue
            v, ai = answers.get(cle), _ai(formulaire, cle)
            origin = ("You" if ai is None else "AI, kept by you" if v == ai else "You, AI corrected")
            rows.append({"Step": title, "Question": nom_champ(cle),
                         "Answer": "Not answered" if v is None else "I'm not sure" if v == "NOT_SURE" else libelle(v),
                         "Answered by": origin})
    rows.append({"Step": "Overall assessment", "Question": "Overall condition",
                 "Answer": libelle(answers["overall"]) if answers.get("overall") else "Not answered",
                 "Answered by": "You"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    kept = sum(r["Answered by"] == "AI, kept by you" for r in rows)
    fixed = sum(r["Answered by"] == "You, AI corrected" for r in rows)
    open_ = sum(r["Answer"] in ("Not answered", "I'm not sure") for r in rows)
    st.caption(f"{kept} AI answer(s) kept, {fixed} corrected by you, {open_} left open.")


def ecran_resultat(a):
    fiche, risque = a["fiche"], a["risque"]
    if st.button("Analyse another photo"):
        st.session_state.pop(f"form_{a['id']}", None)
        del st.session_state["analyse"]
        st.rerun()

    if not fiche.get("valide"):
        st.error(tr(fiche.get("message", "Photo cannot be used.")))
        return

    state = _state(a)
    answers, step = state["answers"], state["step"]
    title, keys = STEPS[step]

    site = (a.get("lieu") or {}).get("site")
    st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)}: {title}")

    col_form, col_photo = st.columns([3, 2], gap="large")
    with col_photo:
        st.image(base64.b64decode(a["image_annotee_b64"]),
                 caption=f"Photo taken facing {ORIENTATIONS.get(fiche.get('orientation'), '?')}: "
                         "water in blue, vegetation in green, artificial surfaces in red")
        if site:
            st.caption(f"Research site: {nom_site(site)}")
        if step == 2:
            st.info("Left and right banks are defined looking downstream.")
        if step in (0, len(STEPS) - 1):
            afficher_risque(risque, detaille=False)
            afficher_signalements(fiche)

    with col_form:
        for cle in keys:
            if cle == "#habitats":
                _checklist(a, "habitats", "Are there any habitats present?", HABITATS, answers)
            elif cle == "#natural_debris":
                _checklist(a, "natural_debris", "Are there any natural debris present?", DEBRIS, answers)
            elif cle == "#water_height":
                _water_height(a, answers)
            elif cle == "#invasive_species":
                _invasive(a, answers)
            elif cle == "#overall":
                _overall(a, answers)
            elif cle == "#feelings":
                _feelings(a, answers)
            elif cle == "#summary":
                _summary(a, answers)
            else:
                _question(a, cle, answers)

        b_prev, b_next = st.columns(2)
        if step > 0 and b_prev.button("Previous", use_container_width=True, key=f"prev_{step}"):
            state["step"] -= 1
            st.rerun()
        if step < len(STEPS) - 1 and b_next.button("Next", type="primary", use_container_width=True,
                                                   key=f"next_{step}"):
            state["step"] += 1
            st.rerun()

        if step == len(STEPS) - 1:
            reponses = {c: answers.get(c) or "NOT_SURE" for c in OPTIONS_ALL}
            reponses.update({k: answers.get(k) for k in ("habitats", "natural_debris", "water_height",
                                                         "invasive_species", "invasive_which", "feelings")})
            formulaire = fiche.get("formulaire", {})
            corrections = {c: {"ia": formulaire.get(c, {}).get("valeur", "NOT_SURE"), "citoyen": v}
                           for c, v in reponses.items() if c in OPTIONS_ALL and _ai(formulaire, c)
                           and v != formulaire.get(c, {}).get("valeur")}
            if a.get("envoyee"):
                st.success("Observation sent. Thank you for your contribution.")
                if risque.get("a_valider_gestionnaire"):
                    st.info("The risk level is high: a manager will check the observation before any alert.")
            elif not answers.get("overall"):
                st.warning("Go back to step 4 and rate the overall condition to send the observation.")
            elif st.button("Send the observation", type="primary", use_container_width=True):
                enregistrer_observation(a, reponses, answers["overall"], corrections)
                st.session_state["analyse"]["envoyee"] = True
                st.rerun()

    if step == len(STEPS) - 1:
        with st.expander("How the AI decided"):
            st.caption("Technical details: confidence and source of each AI answer, risk computation. "
                       "Answers hidden from the form after evaluation still appear here.")
            afficher_formulaire(fiche)
            st.markdown("**Risk score**")
            st.write(f"Humans: score {risque['risque_humains']['score']}, "
                     f"animals: score {risque['risque_animaux']['score']}")
            afficher_contexte(risque, dans_encadre=True)


def page_nouvelle():
    st.title("📷 New observation")
    if "analyse" in st.session_state:
        ecran_resultat(st.session_state["analyse"])
    else:
        ecran_envoi()


# ---------- "Analysed examples" space ----------
def page_exemples(fiches, risques):
    st.title("🌊 Analysed examples")
    st.caption("Photos of the test set, analysed in advance in Colab.")
    photos = [f["photo"] for f in fiches]
    choix = st.selectbox("Photo", photos)
    fiche = next(f for f in fiches if f["photo"] == choix)

    if not fiche.get("valide", True) or "formulaire" not in fiche:
        st.error(tr(fiche.get("message", "This photo cannot be used: the stream is not visible enough.")))
        return

    col_img, col_risque = st.columns([3, 2], gap="large")
    with col_img:
        img = chemin_image(fiche)
        if img:
            st.image(str(img), caption=f"Photo taken facing {ORIENTATIONS.get(fiche.get('orientation'), '?')}: "
                                       "water in blue, vegetation in green, artificial surfaces in red")
        else:
            st.warning(f"Annotated image not found: add {nom(choix)} to demo/annotees/.")
    with col_risque:
        r = risques.get(nom(choix))
        if r:
            afficher_risque(r)
        else:
            st.info("No risk score computed for this photo.")

    st.divider()
    oa = fiche.get("overall_assessment", {})
    c1, c2 = st.columns(2)
    c1.metric("Form pre-filled", f"{fiche.get('taux_prefill', 0):.0%}")
    c2.metric("Suggested overall assessment", libelle(oa.get("valeur")) if oa.get("valeur") else "–")
    if oa and oa.get("accord_modele") is False:
        st.info("The rules and the model disagree on the overall assessment: the citizen decides.")
    if oa.get("raisons"):
        with st.expander("Overall assessment details"):
            for raison in oa["raisons"]:
                st.write("•", tr(raison))

    afficher_signalements(fiche)
    st.subheader("Pre-filled form")
    afficher_formulaire(fiche)
    afficher_questions_laissees(fiche)
    bouton_fhir(choix, fiche, risques.get(nom(choix)))


# ---------- "Manager" space ----------
def page_gestionnaire(fiches, risques):
    st.title("🛡️ Validation queue")
    st.caption("High-risk observations wait for your decision before any alert. "
               "In a real deployment, this space is restricted to authenticated managers.")

    par_photo = {nom(f["photo"]): f for f in fiches}
    dossiers = [(o["id"], o["fiche"], o["risque"], o) for o in lire_observations()]
    dossiers += [(p, par_photo.get(p, {}), r, None) for p, r in risques.items()]

    niveaux = [norm(r["risque_humains"]["niveau"]) for _, _, r, _ in dossiers if r]
    c = st.columns(4)
    c[0].metric("Observations", len(niveaux))
    c[1].metric("Low risk", niveaux.count("FAIBLE"))
    c[2].metric("Moderate risk", niveaux.count("MODERE"))
    c[3].metric("High risk", niveaux.count("ELEVE"))

    decisions = lire_json(DECISIONS, {})
    a_traiter = [d for d in dossiers if d[2] and (
        d[2].get("a_valider_gestionnaire")
        or d[1].get("signalements", {}).get("animal_mort", {}).get("a_valider"))]
    if not a_traiter:
        st.success("No observation waiting.")
        return

    for cle, f, r, obs in a_traiter:
        dec = decisions.get(cle)
        titre = ("New: " if obs else "") + (f"{f.get('photo', cle)}: humans "
                 f"{level(r['risque_humains']['niveau'])}, animals {level(r['risque_animaux']['niveau'])}")
        site_obs = ((obs or {}).get("lieu") or {}).get("site")
        if site_obs:
            titre += f", site {nom_site(site_obs)}"
        if dec:
            titre += f" ({dec['decision']})"
        with st.expander(titre, expanded=dec is None):
            col_img, col_info = st.columns([2, 3])
            img = chemin_image(f)
            if img:
                col_img.image(str(img))
            with col_info:
                if obs:
                    st.caption(f"Sent by a citizen on {obs['date'].replace('T', ' at ')}.")
                for raison in r.get("raisons", []):
                    st.write("•", tr(raison))
                am = f.get("signalements", {}).get("animal_mort", {})
                if am.get("a_valider"):
                    st.error(f"Dead animal reported by the AI: {tr(am.get('raison', ''))}")
                if obs and obs.get("corrections"):
                    st.markdown("**Corrected by the citizen**")
                    for champ, c_ in obs["corrections"].items():
                        st.write(f"• {nom_champ(champ)}: AI {libelle(c_['ia'])}, "
                                 f"citizen {libelle(c_['citoyen'])}")
                b1, b2 = st.columns(2)
                if b1.button("Validate the alert", key=f"ok_{cle}", type="primary"):
                    ecrire_decision(cle, "alert validated")
                    st.rerun()
                if b2.button("Reject", key=f"ko_{cle}"):
                    ecrire_decision(cle, "rejected")
                    st.rerun()
                bouton_fhir(cle, f, r, obs, dec)


# ---------- Launch ----------
try:
    fiches, risques = charger()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run the app from the project root, "
             "with the results in the demo/ folder.")
    st.stop()

page = st.sidebar.radio("Space", ["New observation", "Analysed examples", "Manager"])

if not url_fixe():   # without a fixed address, ask for it in the menu
    with st.sidebar.expander("Analysis connection", expanded=not url_api()):
        st.text_input("API address", key="url_api", placeholder="https://….ngrok-free.app",
                      help="Shown by the last cell of the API_streamsentinel notebook in Colab.")
        if st.button("Test the connection", disabled=not url_api()):
            try:
                s = requests.get(f"{url_api()}/sante", headers=ENTETES, timeout=15).json()
                st.success("Connected" + (" (GPU on)" if s.get("gpu") else " (no GPU: slow analysis)"))
            except (requests.RequestException, ValueError):
                st.error("No answer: is the Colab notebook still running?")

st.sidebar.caption("StreamSentinel · OneAquaHealth IEEE Hackathon 2026, track 3")

if page == "New observation":
    page_nouvelle()
elif page == "Analysed examples":
    page_exemples(fiches, risques)
else:
    page_gestionnaire(fiches, risques)
