"""FHIR R4 export of StreamSentinel observations.

Aligned with the OneAquaHealth implementation guide (hl7-eu/oah, working version
0.1.0-ci-build, https://build.fhir.org/ig/hl7-eu/oah/):
- each question of the citizen form becomes an Observation whose first code is the
  matching OneAquaHealth indicator (Temporary OAH Code System) and whose second code
  is the exact question of the citizen app;
- the research site becomes a Location, identified by its official code (ENORA API);
- status is "preliminary" until a manager validates the observation, "final" afterwards:
  only final observations declare the observation-indicators-oah profile.

The guide is still a draft, so some validator warnings are expected (unpublished
profiles, StreamSentinel code systems not hosted on a terminology server).

Note: the input dictionaries come from the J4/J5 pipeline and keep its French keys
(formulaire, valeur, confiance, risque_humains...).
"""
import html
import unicodedata
import uuid
from datetime import datetime, timezone

OAH_CODES = "http://hl7.eu/fhir/ig/oah/CodeSystem/temporarySystem-oah-eu"
OAH_OBSERVATION_PROFILE = "http://hl7.eu/fhir/ig/oah/StructureDefinition/observation-indicators-oah"
ENORA_SITES = "https://api.enora-oah.eu/api/sites"
SS_BASE = "https://github.com/momo25bend/streamsentinel/fhir/CodeSystem"
SS_QUESTIONS = f"{SS_BASE}/citizen-question"
SS_ANSWERS = f"{SS_BASE}/citizen-answer"
SS_ASSESSMENT = f"{SS_BASE}/assessment"
SS_WORKFLOW = f"{SS_BASE}/workflow-status"
DATA_ABSENT = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
AI_METHOD = "AI-assisted photo analysis (StreamSentinel)"

# Citizen app question -> OneAquaHealth indicator.
# "morophology" is the spelling used by the official guide (a typo reported to its
# authors); it is kept on purpose so records validate against the current guide.
INDICATORS = {
    "channel_form": "morophology", "bottom_type": "morophology", "bank_type": "morophology",
    "water_flow": "hydrology", "barriers": "hydrology",
    "water_aspect": "foam",
    "draining_pipes": "LandUse", "sewage_discharge": "LandUse", "construction": "LandUse",
    "impervious_left": "LandUse", "impervious_right": "LandUse",
    "vegetation_left": "riparianVegetation", "vegetation_right": "riparianVegetation",
    "vegetation_type_left": "riparianVegetation", "vegetation_type_right": "riparianVegetation",
    "vegetation_cuts": "riparianVegetation",
}
INDICATOR_LABELS = {
    "morophology": "Morphology of the streams",
    "hydrology": "Hydrology of the stream",
    "foam": "Foam/colour/smell",
    "LandUse": "Land use in the margins",
    "riparianVegetation": "Riparian vegetation",
}
# Citizen app answers that have a direct equivalent in the OneAquaHealth code system
OAH_VALUES = {
    "TREES": ("trees", "Trees (height >3m)"),
    "SHRUBS": ("bushes", "Bushes (height (1.5-3m)"),
    "HERBS": ("herbaceous", "Herbaceous (height < 1.5m)"),
}
RISK_LEVELS = {"FAIBLE": "low", "MODERE": "moderate", "ELEVE": "high"}
# StreamSentinel answer code -> code stored by the OneAquaHealth citizen app
# (read from the app's step 9 summary screen). Codes not listed here are identical
# in both systems (FLAT, NAT, FAS, GOOD...) or not shown by the app (vegetation types).
APP_ANSWERS = f"{SS_BASE}/oah-app-answer"
APP_CODES = {
    "water_aspect": {"CLEAR": "CL", "MUDDY": "MU", "FOAM": "FO", "ALTERED_COLOR": "CO"},
    "water_flow":   {"FAS": "FAS", "SLOW": "NOR", "STAGNANT": "STA", "DRY": "DRY"},
    "channel_form": {"FLAT": "FLAT", "U_SHAPE": "U", "V_SHAPE": "V"},
    "bottom_type":  {"NAT": "NAT", "ARTIFICIAL": "ART"},
    "bank_type":    {"NAT": "NAT", "ARTIFICIAL": "ART", "LAYED_STONES": "LAS"},
}
# Follow-up questions the app only asks when the parent answer is YES
CONDITIONAL = {"vegetation_type_left": "vegetation_left", "vegetation_type_right": "vegetation_right"}


def to_app_code(question, value):
    """Return the OneAquaHealth app code for an answer, or the answer unchanged."""
    return APP_CODES.get(question, {}).get(value, value)


def _narrative(summary):
    """Human-readable summary recommended by FHIR (constraint dom-6)."""
    return {"status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml"><p>{html.escape(summary)}</p></div>'}


def _urn():
    return f"urn:uuid:{uuid.uuid4()}"


def _strip_accents(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().upper()


def _location(place):
    site = (place or {}).get("site")
    resource = {"resourceType": "Location", "status": "active", "mode": "instance"}
    if site:
        resource["name"] = " - ".join(v for v in (site.get("ville"), site.get("nom")) if v) or site.get("code")
        resource["identifier"] = [{"system": ENORA_SITES, "value": site.get("code")}]
    else:
        resource["name"] = "Observation point outside the OneAquaHealth research network"
    if place and place.get("lat") is not None:
        resource["position"] = {"latitude": round(float(place["lat"]), 6),
                                "longitude": round(float(place["lon"]), 6)}
    summary = f"Urban stream reach: {resource['name']}"
    if "position" in resource:
        summary += f" ({resource['position']['latitude']}, {resource['position']['longitude']})"
    resource["text"] = _narrative(summary)
    return resource


def _observation(location_ref, date, status, code, value=None, absent=False, note=None, method=None):
    obs = {
        "resourceType": "Observation",
        "status": status,
        "category": [{"text": "Citizen science stream assessment"}],
        "code": code,
        "subject": {"reference": location_ref},
        "effectiveDateTime": date,
        "performer": [{"display": "Citizen scientist (anonymous)"}],
    }
    if status == "final":
        obs["meta"] = {"profile": [OAH_OBSERVATION_PROFILE]}
    if absent:
        obs["dataAbsentReason"] = {"coding": [{"system": DATA_ABSENT, "code": "asked-unknown",
                                               "display": "Asked But Unknown"}]}
    elif value is not None:
        obs["valueCodeableConcept"] = value
    if method:
        obs["method"] = {"text": method}
    if note:
        obs["note"] = [{"text": note}]
    if absent:
        result = "unknown (to be completed on site)"
    else:
        result = (value or {}).get("text") or ", ".join(
            c.get("display") or c.get("code", "") for c in (value or {}).get("coding", []))
    obs["text"] = _narrative(f"{code.get('text', 'Observation')}: {result} [{status}]")
    return obs


def record_to_fhir(record, risk=None, answers=None, rating=None, place=None,
                   validated=False, date=None):
    """Convert a StreamSentinel record into a FHIR R4 Bundle (type "collection").

    record    : J4 record (formulaire, overall_assessment, signalements...)
    risk      : J5 record (risque_humains, risque_animaux, raisons)
    answers   : final citizen answers {question: code}; defaults to the AI proposals
    rating    : overall rating chosen by the citizen; defaults to the AI proposal
    place     : {"lat", "lon", "site"} picked on the map
    validated : True once a manager has validated the observation (status "final")
    """
    status = "final" if validated else "preliminary"
    date = date or datetime.now(timezone.utc).isoformat(timespec="seconds")
    form = record.get("formulaire", {})
    answers = answers or {key: item.get("valeur") for key, item in form.items()}

    entries = []
    location_ref = _urn()
    entries.append({"fullUrl": location_ref, "resource": _location(place)})

    def add(resource):
        entries.append({"fullUrl": _urn(), "resource": resource})

    # 1. One Observation per form question
    for key, answer in answers.items():
        indicator = INDICATORS.get(key)
        if not indicator:
            continue
        parent = CONDITIONAL.get(key)
        if parent and answers.get(parent) != "YES":
            continue  # the app does not ask this question, so it is not exported
        code = {"coding": [
                    {"system": OAH_CODES, "code": indicator, "display": INDICATOR_LABELS[indicator]},
                    {"system": SS_QUESTIONS, "code": key}],
                "text": key.replace("_", " ")}
        ai = form.get(key, {})
        provenance = f"AI proposal: {ai.get('valeur', 'NOT_SURE')}"
        if ai.get("valeur") not in (None, "NOT_SURE"):
            provenance += f" (confidence {ai.get('confiance', 0):.0%}, source {ai.get('source', '?')})"
        provenance += f"; recorded answer: {answer or 'NOT_SURE'}"

        if answer in (None, "NOT_SURE"):
            add(_observation(location_ref, date, status, code, absent=True,
                             note=provenance, method=AI_METHOD))
            continue
        codings = [{"system": SS_ANSWERS, "code": answer}]
        app_code = to_app_code(key, answer)
        if app_code != answer:
            codings.append({"system": APP_ANSWERS, "code": app_code})
        if answer in OAH_VALUES:
            oah_code, display = OAH_VALUES[answer]
            codings.insert(0, {"system": OAH_CODES, "code": oah_code, "display": display})
        add(_observation(location_ref, date, status, code, value={"coding": codings},
                         note=provenance, method=AI_METHOD))

    # 2. Overall assessment (Good / Moderate / Poor, as in the citizen app)
    overall = record.get("overall_assessment", {})
    final_rating = rating or overall.get("valeur")
    if final_rating:
        add(_observation(
            location_ref, date, status,
            {"coding": [{"system": SS_ASSESSMENT, "code": "overall-assessment"}], "text": "Overall assessment"},
            value={"coding": [{"system": SS_ANSWERS, "code": final_rating}]},
            note=f"AI proposal: {overall.get('valeur', '?')}; recorded rating: {final_rating}"))

    # 3. One Health exposure risk (indicative only)
    for key, group in (("risque_humains", "humans"), ("risque_animaux", "animals")):
        block = (risk or {}).get(key)
        if not block:
            continue
        level = RISK_LEVELS.get(_strip_accents(block.get("niveau")), "unknown")
        reasons = "; ".join((risk or {}).get("raisons", []))
        add(_observation(
            location_ref, date, status,
            {"coding": [{"system": SS_ASSESSMENT, "code": f"exposure-risk-{group}"}],
             "text": f"Indicative exposure risk for {group}"},
            value={"coding": [{"system": SS_ASSESSMENT, "code": level}], "text": level},
            note=f"Indicative estimate, not official health advice. Score {block.get('score')}. {reasons}",
            method="Rule-based score: photo analysis + Open-Meteo + Hub'Eau"))

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": date,
        "meta": {"tag": [{"system": SS_WORKFLOW,
                          "code": "validated-by-manager" if validated else "awaiting-validation"}]},
        "entry": entries,
    }
