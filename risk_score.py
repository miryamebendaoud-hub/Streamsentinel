"""Risk score for people and animals (J5), without any AI model.

Same code as notebooks/API_streamsentinel.ipynb: weather from Open-Meteo, river flow from Hub'Eau,
rules traced to the OneAquaHealth factsheets. It only needs the pre-filled form and the place,
so the app can compute it live for any location, even when the GPU is off.
"""
import requests


# ---- notebook cell 22 ----
import requests

def meteo_contexte(lat, lon):
    """Pluie et température : 3 jours passés + 3 jours de prévision."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max",
        "past_days": 3,
        "forecast_days": 4,   # aujourd'hui + 3 jours
        "timezone": "Europe/Paris",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
    except Exception as e:
        return {"disponible": False, "erreur": str(e)}

    pluie = [p or 0 for p in d["precipitation_sum"]]
    temp = [t for t in d["temperature_2m_max"] if t is not None]
    # indices : 0-2 = 3 jours passés, 3 = aujourd'hui, 4-6 = 3 jours à venir
    return {
        "disponible": True,
        "pluie_72h_mm": round(sum(pluie[0:3]), 1),
        "pluie_aujourdhui_mm": round(pluie[3], 1),
        "pluie_prevue_3j_mm": round(sum(pluie[4:7]), 1),
        "temp_max_aujourdhui": d["temperature_2m_max"][3],
        "temp_max_3j": max(temp[3:7]) if temp else None,
    }


# ---- notebook cell 23 ----
def val(fiche, champ):
    """Valeur d'un champ du formulaire, en liste (gère simple et multiple)."""
    v = fiche["formulaire"].get(champ, {}).get("valeur")
    return v if isinstance(v, list) else [v]

def niveau(score):
    return "ELEVE" if score >= 4 else "MODERE" if score >= 2 else "FAIBLE"

CONSIGNES = {
    "humains": {
        "FAIBLE": "Pas de danger visible. Évitez tout de même de boire l'eau.",
        "MODERE": "Évitez le contact avec l'eau et lavez-vous les mains après la visite.",
        "ELEVE":  "Ne touchez pas l'eau. Signalement transmis au gestionnaire pour vérification.",
    },
    "animaux": {
        "FAIBLE": "Pas de danger visible pour les animaux.",
        "MODERE": "Empêchez votre chien de boire ou de se baigner.",
        "ELEVE":  "Tenez les animaux en laisse, loin de l'eau. Signalement transmis au gestionnaire.",
    },
}

def evaluer_risque(fiche, meteo):
    h, a, raisons = 0, 0, []   # score humains, score animaux
    aspect = val(fiche, "water_aspect")
    chaud = meteo.get("disponible") and (meteo.get("temp_max_aujourdhui") or 0) >= 25
    pluie = meteo.get("pluie_72h_mm", 0) if meteo.get("disponible") else 0

    # --- Signaux de l'image (J1 à J4) ---
    if "FOAM" in aspect:
        h += 2; a += 2; raisons.append("mousse sur l'eau : pollution possible")
    if "ALTERED_COLOR" in aspect:
        h += 2; a += 2; raisons.append("couleur de l'eau anormale")
        if chaud:
            h += 1; a += 2; raisons.append("eau colorée + chaleur : cyanobactéries possibles, dangereuses pour les chiens")
    if "MUDDY" in aspect:
        h += 1; a += 1; raisons.append("eau trouble")
    if "YES" in val(fiche, "sewage_discharge"):
        h += 3; a += 3; raisons.append("rejet d'eaux usées visible")
    if "YES" in val(fiche, "draining_pipes"):
        h += 1; a += 1; raisons.append("canalisation de drainage visible")

    sig = fiche.get("signalements", {})
    if sig.get("animal_mort", {}).get("valeur") == "YES":
        h += 3; a += 3; raisons.append("animal mort signalé (à confirmer)")
    if sig.get("dechets"):
        h += 1; a += 1; raisons.append(f"{len(sig['dechets'])} déchet(s) détecté(s) : blessure ou ingestion")

    # --- Contexte météo ---
    if pluie >= 20:
        h += 2; a += 2; raisons.append(f"{pluie} mm de pluie en 72 h : ruissellement et débordements d'égouts probables")
    elif pluie >= 10:
        h += 1; a += 1; raisons.append(f"{pluie} mm de pluie en 72 h")

    flow = str(val(fiche, "water_flow")[0])
    if flow.startswith("FAS") and pluie >= 10:
        h += 1; raisons.append("courant rapide après la pluie : risque de chute ou d'emport")
    if flow == "STAGNANT" and chaud:
        a += 1; raisons.append("eau stagnante et chaude")

    nh, na = niveau(h), niveau(a)

    # --- Tendance à 3 jours (optionnelle) ---
    tendance = "inconnue"
    if meteo.get("disponible"):
        tendance = "en hausse" if meteo["pluie_prevue_3j_mm"] >= 20 else "stable"

    return {
        "risque_humains": {"niveau": nh, "score": h, "consigne": CONSIGNES["humains"][nh]},
        "risque_animaux": {"niveau": na, "score": a, "consigne": CONSIGNES["animaux"][na]},
        "raisons": raisons or ["aucun signal de risque détecté"],
        "tendance_3j": tendance,
        "meteo": meteo,
        "a_valider_gestionnaire": "ELEVE" in (nh, na),  # jamais d'alerte directe
    }

# ---- notebook cell 24 ----
from datetime import date, timedelta
from math import radians, sin, cos, asin, sqrt

HUBEAU_URL = "https://hubeau.eaufrance.fr/api"

def hubeau(chemin, params):
    """Essaie l'API v2 puis v1 ; renvoie la liste 'data' ou None."""
    for v in ("v2", "v1"):
        try:
            r = requests.get(f"{HUBEAU_URL}/{v}/hydrometrie/{chemin}", params=params, timeout=20)
            if r.status_code in (200, 206):
                return r.json().get("data", [])
        except Exception:
            pass
    return None

def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

def hubeau_contexte(lat, lon, rayon_km=10):
    stations = hubeau("referentiel/stations",
                      {"latitude": lat, "longitude": lon, "distance": rayon_km, "size": 20})
    if not stations:
        return {"disponible": False, "raison": f"aucune station à moins de {rayon_km} km"}

    # Trier par distance, garder la première station qui mesure un débit
    stations.sort(key=lambda s: distance_km(lat, lon, s["latitude_station"], s["longitude_station"]))
    for s in stations:
        code = s["code_station"]
        obs = hubeau("observations_tr", {"code_entite": code, "grandeur_hydro": "Q",
                                         "size": 1, "sort": "desc"})
        if not obs:
            continue
        debit = obs[0]["resultat_obs"] / 1000   # l/s → m³/s

        # Référence : débits moyens journaliers des 30 derniers jours
        elab = hubeau("obs_elab", {"code_entite": code, "grandeur_hydro_elab": "QmnJ",
                                   "date_debut_obs_elab": str(date.today() - timedelta(days=30)),
                                   "size": 40}) or []
        valeurs = sorted(e["resultat_obs_elab"] / 1000 for e in elab if e.get("resultat_obs_elab"))
        mediane = valeurs[len(valeurs)//2] if valeurs else None

        return {
            "disponible": True,
            "station": s.get("libelle_station"),
            "code_station": code,
            "distance_km": round(distance_km(lat, lon, s["latitude_station"], s["longitude_station"]), 1),
            "debit_m3s": round(debit, 2),
            "debit_median_30j_m3s": round(mediane, 2) if mediane else None,
            "ratio": round(debit / mediane, 2) if mediane else None,
        }
    return {"disponible": False, "raison": "stations proches sans mesure de débit"}


# ---- notebook cell 25 ----
def appliquer_debit(r, hub):
    """Ajoute les règles Hub'Eau à un résultat de evaluer_risque."""
    h, a = r["risque_humains"]["score"], r["risque_animaux"]["score"]
    ratio = hub.get("ratio") if hub.get("disponible") else None

    if ratio is not None:
        if ratio >= 2:
            h += 2; a += 1
            r["raisons"].append(f"débit {ratio} fois supérieur à la normale : crue, courant dangereux")
        elif ratio >= 1.5:
            h += 1
            r["raisons"].append(f"débit élevé ({ratio} fois la normale)")
        elif ratio <= 0.3:
            a += 1
            r["raisons"].append(f"débit très faible ({ratio} fois la normale) : pollution plus concentrée")

    nh, na = niveau(h), niveau(a)
    r["risque_humains"] = {"niveau": nh, "score": h, "consigne": CONSIGNES["humains"][nh]}
    r["risque_animaux"] = {"niveau": na, "score": a, "consigne": CONSIGNES["animaux"][na]}
    r["a_valider_gestionnaire"] = "ELEVE" in (nh, na)
    r["hubeau"] = hub
    if r["raisons"][0] == "aucun signal de risque détecté" and len(r["raisons"]) > 1:
        r["raisons"].pop(0)
    return r

def evaluer_risque_complet(fiche, lat, lon, meteo=None, hub=None):
    meteo = meteo or meteo_contexte(lat, lon)
    hub = hub or hubeau_contexte(lat, lon)
    return appliquer_debit(evaluer_risque(fiche, meteo), hub)


# ---- sourced risk rules (same scores) ----
# ===== J5 bis : chaque règle du risque rattachée à sa source =====
# Les fiches OneAquaHealth (Schmeller et al., 2026, doi:10.5281/zenodo.20345207, CC-BY 4.0)
# justifient POURQUOI un signal compte pour la santé ; elles ne donnent AUCUN seuil
# chiffré pour des observations photo. Les points et les seuils restent donc des
# choix de l'équipe, affichés comme tels. Les scores sont identiques à la version précédente.
SOURCES_OAH = {
    "I":    "OAH factsheet I (diatoms): harmful algal blooms produce toxins, kill fish and threaten human health",
    "III":  "OAH factsheet III (fish): fish can carry pathogens transmissible to humans",
    "VII":  "OAH factsheet VII (fecal coliforms): waste inputs mean exposure to enteric pathogens",
    "VIII": "OAH factsheet VIII (pathogens): abundance rises with wastewater discharge and runoff",
    "IX":   "OAH factsheet IX (antibiotic resistance genes): spread through wastewater and urban runoff",
    "EQUIPE": "Team assumption: not covered by the OAH factsheets",
}

def _regle(detail, raisons, texte, dh, da, sources, label_en):
    detail.append({"texte": texte, "label": label_en, "humains": dh, "animaux": da,
                   "sources": [SOURCES_OAH[s] for s in sources]})
    raisons.append(texte)
    return dh, da

def evaluer_risque(fiche, meteo):
    h, a, raisons, detail = 0, 0, [], []
    aspect = val(fiche, "water_aspect")
    chaud = meteo.get("disponible") and (meteo.get("temp_max_aujourdhui") or 0) >= 25
    pluie = meteo.get("pluie_72h_mm", 0) if meteo.get("disponible") else 0

    def r(texte, dh, da, sources, label):
        nonlocal h, a
        x, y = _regle(detail, raisons, texte, dh, da, sources, label)
        h += x; a += y

    if "FOAM" in aspect:
        r("mousse sur l'eau : pollution possible", 2, 2, ["EQUIPE"], "Foam on the water: possible pollution")
    if "ALTERED_COLOR" in aspect:
        r("couleur de l'eau anormale", 2, 2, ["I"], "Abnormal water colour: possible algal bloom")
        if chaud:
            r("eau colorée + chaleur : cyanobactéries possibles, dangereuses pour les chiens", 1, 2,
              ["EQUIPE"], "Coloured water and heat (25 °C or more): bloom more likely")
    if "MUDDY" in aspect:
        r("eau trouble", 1, 1, ["EQUIPE"], "Muddy water")
    if "YES" in val(fiche, "sewage_discharge"):
        r("rejet d'eaux usées visible", 3, 3, ["VII", "VIII", "IX"], "Visible sewage discharge")
    if "YES" in val(fiche, "draining_pipes"):
        r("canalisation de drainage visible", 1, 1, ["IX"], "Visible drainage pipe (urban runoff)")

    sig = fiche.get("signalements", {})
    if sig.get("animal_mort", {}).get("valeur") == "YES":
        r("animal mort signalé (à confirmer)", 3, 3, ["III", "I"], "Dead animal reported (to be confirmed)")
    if sig.get("dechets"):
        n = len(sig["dechets"])
        r(f"{n} déchet(s) détecté(s) : blessure ou ingestion", 1, 1, ["EQUIPE"],
          f"{n} litter item(s): injury or ingestion")

    if pluie >= 20:
        r(f"{pluie} mm de pluie en 72 h : ruissellement et débordements d'égouts probables", 2, 2,
          ["VIII", "IX"], f"{pluie} mm of rain in 72 h: runoff and sewer overflows (threshold: team)")
    elif pluie >= 10:
        r(f"{pluie} mm de pluie en 72 h", 1, 1, ["VIII", "IX"], f"{pluie} mm of rain in 72 h (threshold: team)")

    flow = str(val(fiche, "water_flow")[0])
    if flow.startswith("FAS") and pluie >= 10:
        r("courant rapide après la pluie : risque de chute ou d'emport", 1, 0, ["EQUIPE"],
          "Fast current after rain: risk of falling in")
    if flow == "STAGNANT" and chaud:
        r("eau stagnante et chaude", 0, 1, ["EQUIPE"], "Stagnant, warm water")

    nh, na = niveau(h), niveau(a)
    tendance = "inconnue"
    if meteo.get("disponible"):
        tendance = "en hausse" if meteo["pluie_prevue_3j_mm"] >= 20 else "stable"
    return {
        "risque_humains": {"niveau": nh, "score": h, "consigne": CONSIGNES["humains"][nh]},
        "risque_animaux": {"niveau": na, "score": a, "consigne": CONSIGNES["animaux"][na]},
        "raisons": raisons or ["aucun signal de risque détecté"],
        "regles": detail,
        "tendance_3j": tendance,
        "meteo": meteo,
        "a_valider_gestionnaire": "ELEVE" in (nh, na),
    }

def appliquer_debit(r, hub):
    h, a = r["risque_humains"]["score"], r["risque_animaux"]["score"]
    ratio = hub.get("ratio") if hub.get("disponible") else None
    regle = None
    if ratio is not None:
        if ratio >= 2:
            regle = (f"débit {ratio} fois supérieur à la normale : crue, courant dangereux", 2, 1,
                     f"River flow {ratio}× the 30-day median: flood, dangerous current")
        elif ratio >= 1.5:
            regle = (f"débit élevé ({ratio} fois la normale)", 1, 0, f"High river flow ({ratio}× normal)")
        elif ratio <= 0.3:
            regle = (f"débit très faible ({ratio} fois la normale) : pollution plus concentrée", 0, 1,
                     f"Very low river flow ({ratio}× normal): pollution more concentrated")
    if regle:
        texte, dh, da, label = regle
        r.setdefault("regles", [])
        _regle(r["regles"], r["raisons"], texte, dh, da, ["EQUIPE"], label)
        h += dh; a += da

    nh, na = niveau(h), niveau(a)
    r["risque_humains"] = {"niveau": nh, "score": h, "consigne": CONSIGNES["humains"][nh]}
    r["risque_animaux"] = {"niveau": na, "score": a, "consigne": CONSIGNES["animaux"][na]}
    r["a_valider_gestionnaire"] = "ELEVE" in (nh, na)
    r["hubeau"] = hub
    if r["raisons"][0] == "aucun signal de risque détecté" and len(r["raisons"]) > 1:
        r["raisons"].pop(0)
    return r

