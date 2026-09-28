"""English display of texts produced by the StreamSentinel pipeline.

The J4/J5 notebooks write their explanations in French (safety advice, risk reasons,
justifications). The app translates them at display time with `tr()`: exact phrases
first, then patterns with numbers. Unknown texts are shown unchanged.

Run `python i18n.py` from the project root to list the texts of demo/*.json that
are not translated yet.
"""
import json
import re
import sys
from pathlib import Path

EXACT = {
    # API progress steps
    "En file d'attente": "Queued",
    "Segmentation de la scène": "Segmenting the scene",
    "Lecture de la photo par le modèle vision-langage": "Reading the photo with the vision-language model",
    "Arbitrage : écume naturelle ou mousse de pollution ?": "Checking: natural whitewater or pollution foam?",
    "Recherche de déchets et d'animaux morts": "Looking for litter and dead animals",
    "Annotation de l'image": "Annotating the image",
    "Météo et débit de la rivière": "Weather and river flow",
    "Analyse terminée": "Analysis complete",
    # Photo rejected by the quality check
    "Cours d'eau pas assez visible : cadrez l'eau et les berges, puis reprenez la photo.":
        "Stream not visible enough: frame the water and both banks, then take the photo again.",
    # Questions left to the citizen
    "choix du site officiel": "choice of the official site",
    "mesure physique, impossible sur photo": "physical measurement, impossible from a photo",
    "pompage rarement visible": "pumping is rarely visible",
    "sous-options non relevées dans l'app": "sub-options not recorded by the app",
    "liste d'espèces non relevée dans l'app": "species list not recorded by the app",
    "ressenti personnel : l'IA ne le remplit jamais": "personal feeling: the AI never fills it in",
    # Justifications
    "réponse du modèle hors options": "model answer outside the allowed options",
    "mousse ou écume naturelle : à confirmer": "foam or natural whitewater: to be confirmed",
    # Risk (J5)
    "aucun signal de risque détecté": "no risk signal detected",
    "animal mort signalé (à confirmer)": "dead animal reported (to be confirmed)",
    "couleur de l'eau anormale": "abnormal water colour",
    "Pas de danger visible. Évitez tout de même de boire l'eau.":
        "No visible danger. Still, do not drink the water.",
    "Pas de danger visible pour les animaux.": "No visible danger for animals.",
    "stable": "stable",
    # Overall assessment reasons
    "berges artificielles": "artificial banks",
    "eau claire": "clear water",
    "rive gauche imperméable": "impervious left bank",
    "rive droite imperméable": "impervious right bank",
    "pas de végétation rive gauche": "no vegetation on the left bank",
    "pas de végétation rive droite": "no vegetation on the right bank",
}

PATTERNS = [
    (r"^(\d+)% de berge artificielle ; confirmé par le modèle$", "{0}% artificial bank; confirmed by the model"),
    (r"^(\d+)% de berge artificielle$", "{0}% artificial bank"),
    (r"^(\d+)% de surfaces artificielles$", "{0}% artificial surfaces"),
    (r"^(\d+)% de la végétation$", "{0}% of the vegetation"),
    (r"^(\d+)% de végétation$", "{0}% vegetation"),
    (r"^turbidité ([\d.]+)$", "turbidity {0}"),
    (r"^(\d+) déchet\(s\) détecté\(s\) : blessure ou ingestion$", "{0} litter item(s) detected: injury or ingestion risk"),
    (r"^le modèle vision-langage propose (\w+) : à vérifier par le citoyen$",
     "the vision-language model suggests {0}: to be checked by the citizen"),
]

LEVELS = {"FAIBLE": "LOW", "MODERE": "MODERATE", "ELEVE": "HIGH"}
SOURCES = {"vlm": "AI model", "regle": "rule", "regle+vlm": "rule + AI model",
           "regle+arbitrage": "rule + AI check", "arbitrage": "AI check"}
ORIENTATIONS = {"aval": "downstream", "amont": "upstream"}


def tr(text):
    """Translate one pipeline text; unknown texts are returned unchanged."""
    if not isinstance(text, str) or not text:
        return text
    text = text.strip()
    if text in EXACT:
        return EXACT[text]
    for pattern, template in PATTERNS:
        m = re.match(pattern, text)
        if m:
            return template.format(*m.groups())
    # "-2 berges artificielles", "+2 eau claire" (overall assessment points)
    m = re.match(r"^([+-]\d+) (.+)$", text)
    if m and m.group(2) in EXACT:
        return f"{m.group(1)} {EXACT[m.group(2)]}"
    # "... ; blanc = écume naturelle selon le modèle"
    if text.endswith(" ; blanc = écume naturelle selon le modèle"):
        return tr(text[: -len(" ; blanc = écume naturelle selon le modèle")]) + "; white = natural whitewater (model)"
    return text


def level(value):
    """FAIBLE / MODÉRÉ / ÉLEVÉ -> LOW / MODERATE / HIGH."""
    import unicodedata
    key = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().upper()
    return LEVELS.get(key, str(value))


def _strings(obj, keys=("raison", "raisons", "consigne", "message", "tendance_3j", "laisses_au_citoyen")):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                if isinstance(v, str):
                    yield v
                elif isinstance(v, list):
                    yield from (x for x in v if isinstance(x, str))
                elif isinstance(v, dict):
                    yield from (x for x in v.values() if isinstance(x, str))
            else:
                yield from _strings(v, keys)
    elif isinstance(obj, list):
        for x in obj:
            yield from _strings(x, keys)


if __name__ == "__main__":
    demo = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    missing = set()
    for f in demo.glob("*.json"):
        for s in _strings(json.loads(f.read_text(encoding="utf-8"))):
            if tr(s) == s.strip() and re.search(r"[éèêàùç]|\b(de|la|le|du|des|rive|eau|pas)\b", s):
                missing.add(s.strip())
    print(f"{len(missing)} untranslated text(s):")
    for s in sorted(missing):
        print(" -", s)
