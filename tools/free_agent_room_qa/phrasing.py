"""Bilingual variants of the page-level operations supported by the room reader."""
import re

from .reader import ROOM_CODE


TAG_COUNT = (
    r'(?:please (?:give|tell)(?: me)? (?:the (?:number|count) of|how many)|how many|count (?:the|all)|'
    r'what is the (?:number|count) of) (?:distinct |unique |different |printed )*(?:door (?:tags|marks|identifiers|ids))'
    r'(?: (?:are |appear |can be seen |are listed |are printed |are shown )?(?:on|in) (?:the|this) '
    r'(?:drawing|floor plan|plan|sheet))?',
    r'(?:wie viele|nenne (?:mir )?die (?:anzahl|zahl) der|zähle (?:die|alle)) '
    r'(?:(?:verschiedene[nr]?|unterschiedliche[nr]?|gedruckte[nr]?) )*türkennungen'
    r'(?: (?:stehen|siehst du|finden sich|sind (?:gedruckt|verzeichnet)) (?:auf|in) '
    r'(?:dem|diesem|der|dieser) (?:blatt|plan|zeichnung|grundriss))?',
)
TITLE = (
    r'(?:what (?:name|title) does (?:the|this) (?:title block|drawing|plan) (?:give|show for|use for) '
    r'(?:the|this) project|what(?: is|[’\x27]s) (?:the )?(?:project name|name of the project) '
    r'(?:on|in) (?:this|the) (?:drawing|sheet|title block|plan))',
    r'(?:welchen (?:projektnamen|projekttitel) (?:trägt|zeigt) (?:diese zeichnung|dieses blatt|dieser plan)|'
    r'wie heißt das projekt (?:im (?:schriftfeld|plankopf)|auf (?:dem|diesem) (?:blatt|plan)))',
)
AREA = (
    r'(?:for @,? (?:read me|give me|what is) (?:the|its) (?:labeled|labelled|printed) (?:floor )?area'
    r'(?: in square (?:metres|meters))?|(?:read|give me) the (?:printed|labeled|labelled) (?:floor )?area '
    r'(?:of|for) (?:room )?@)',
    r'(?:wie groß ist (?:raum )?@ laut (?:der )?(?:(?:aufgedruckt|eingetragen|angegeben)(?:er|en|e))?\s*'
    r'flächenangabe(?: in quadratmetern)?|welche (?:grund)?fläche steht (?:im plan )?(?:bei|für) (?:raum )?@)',
)
CONNECTION = (
    r'(?:through (?:which|what) (?:tagged |marked |labelled |labeled )?doors? can (?:you|i|we) '
    r'(?:walk|go|pass) (?:straight |directly )?from (?:room )?@ (?:into|to) (?:room )?@|'
    r'which (?:marked |tagged )?door(?:s| ids)? (?:lead|leads|open|opens) (?:directly )?from (?:room )?@ (?:into|to) (?:room )?@)',
    r'(?:über welche (?:gekennzeichneten |beschrifteten )?türen gelangt man (?:unmittelbar |direkt )?von (?:raum )?@ '
    r'(?:nach|in) (?:raum )?@|durch welche tür(?:en)? (?:kommt|gelangt) man (?:direkt )?von (?:raum )?@ (?:nach|in) (?:raum )?@)',
)


def alternative(question):
    text = question.lower()
    rooms = [s.upper() for s in ROOM_CODE.findall(question)]
    form = ROOM_CODE.sub('@', text)
    def matches(patterns):
        return any(re.fullmatch(pattern, form) for pattern in patterns)
    if len(rooms) == 2 and rooms[0] != rooms[1] and matches(CONNECTION):
        return 'adjacent_doors', {'rooms':rooms}
    if len(rooms) == 1 and matches(AREA):
        return 'printed_area', {'room':rooms[0]}
    if not rooms and matches(TAG_COUNT):
        return 'count_marks', {}
    if not rooms and matches(TITLE):
        return 'project_title', {}
    return 'unsupported', {}
