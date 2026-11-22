"""Recognize ordinary questions that ask to locate or measure a room."""
import re


ROOMS = (
    ("hallway", "corridor", "passage", "flur", "korridor", "gang", "verbindungsflur"),
    ("lobby", "foyer", "eingangshalle"),
    ("kitchen", "küche"),
    ("bedroom", "schlafzimmer"),
    ("office", "büro"),
    ("bathroom", "bad", "badezimmer", "wc", "toilet"),
    ("living room", "wohnzimmer"),
    ("gym", "fitness room", "fitnessraum"),
    ("hall", "multi-purpose hall", "multipurpose hall", "mehrzweckhalle"),
    ("classroom", "klassenraum", "klassenzimmer"),
    ("storage room", "lagerraum", "abstellraum"),
)
NAMES = {name: group for group in ROOMS for name in group}
NOUN = "|".join(sorted((re.escape(name) for name in NAMES), key=len, reverse=True))
TARGET = (
    rf"(?:(?:the |a |den |die |das |der |des |dem )?(?:(?:main|central|connecting|upper|lower|"
    rf"mittleren|oberen|unteren|großen|kleinen) )?(?:{NOUN})(?: [0-9][a-z0-9-]*)?|"
    r"(?:room |raum )?r[-.]?[0-9]{1,4})"
)
WHERE = (
    r"(?:mark|highlight|outline|find|locate|show|point (?:out|to)|where is|"
    r"markiere?|zeige?|finde|umrande|wo ist|wo liegt|zeichne)"
)
LOCATE = re.compile(rf"{WHERE} (?:me |mir )?(?P<room>{TARGET})(?: ein)?")
AREA = re.compile(
    rf"(?:how (?:big|large) is |what(?: is|'s) (?:the )?(?:(?:floor |net )?area|size) of |"
    rf"(?:measure|calculate|estimate) (?:the )?(?:floor )?area of |"
    rf"how many square (?:metres|meters|feet) (?:is|does) |"
    rf"wie gro(?:ß|ss) ist |wie viel(?:e)? (?:quadratmeter|m2) hat |"
    rf"was ist (?:die )?(?:boden|grund)?fläche von |(?:miss|berechne) (?:die )?(?:boden|grund)?fläche von )"
    rf"(?P<room>{TARGET})(?: (?:have|groß))?"
)
ANAPHORA = re.compile(
    r"(?:how (?:big|large) (?:is it|it is)|what(?: is|'s) its (?:floor )?(?:area|size)|"
    r"how many square (?:metres|meters|feet) (?:is it|it is)|"
    r"(?:measure|calculate|estimate|give me) (?:its|the) (?:floor )?area|"
    r"wie gro(?:ß|ss) (?:ist (?:er|sie|es)|(?:er|sie|es) ist)|"
    r"wie viel(?:e)? (?:quadratmeter|m2) (?:hat (?:er|sie|es)|(?:er|sie|es) hat)|"
    r"(?:miss|berechne) (?:seine|ihre|die) (?:boden|grund)?fläche)"
)
GERMAN = re.compile(r"\b(?:markiere?|zeige?|wo|finde|miss|zeichne|berechne|wie viel|wie groß|quadratmeter)\b")


def clean_target(target):
    return re.sub(r"^(?:the|a|den|die|das|der|des|dem|room|raum) ", "", target)


def compile_room_question(question):
    q = question.lower().replace("’", "'")
    q = re.sub(r"^(?:please|bitte|can you|could you|kannst du(?: mir)?|könntest du(?: mir)?) ", "", q)
    q = re.sub(r"[?!., ]+$", "", q)
    ft = re.search(r"\b(?:square feet|sq\.? ft|ft2)\b", q)
    q = re.sub(r"[, ]+in (?:square (?:metres|meters|feet)|quadratmetern|m2|sq\.? ft|ft2)$", "", q)
    q = re.sub(r" (?:on (?:the|this) (?:sheet|plan|drawing)|im (?:plan|grundriss))$", "", q)
    q = re.sub(r"\s*(?:,?\s+(?:and|und)\s+|,\s*)(?:tell me|sag(?:e)? mir)\s*,?\s*", ", ", q)
    parts = re.split(r"\s*(?:[,;?]|\band\b|\bund\b)\s*", q)
    locate = LOCATE.fullmatch(parts[0])
    area = AREA.fullmatch(parts[0])
    if len(parts) > 1:
        if len(parts) != 2 or not locate:
            return None
        followup = AREA.fullmatch(parts[1])
        if not ANAPHORA.fullmatch(parts[1]) and (
            not followup or clean_target(followup["room"]) != clean_target(locate["room"])
        ):
            return None
        op, target = "measure_room", locate["room"]
    elif area:
        op, target = "measure_room", area["room"]
    elif locate:
        op, target = "locate_room", locate["room"]
    else:
        return None
    return {"op": op, "args": {"room": clean_target(target), "unit": "ft²" if ft else "m²",
                               "lang": "de" if GERMAN.search(question.lower()) else "en"}}
