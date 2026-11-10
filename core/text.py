"""OCR preprocessing from 09a cell 10, unchanged."""
import re
import unicodedata
import pandas as pd

SLANG_DICT = {
    "gw":"saya","gue":"saya","gua":"saya","guwe":"saya","aq":"aku","akoh":"aku","akuh":"aku",
    "lu":"kamu","lo":"kamu","loe":"kamu","elo":"kamu","elu":"kamu","kmu":"kamu","kamuh":"kamu",
    "ane":"saya","ente":"kamu","agan":"kamu","sis":"kakak","bro":"kawan","bray":"kawan",
    "bre":"kawan","cuy":"kawan","klian":"kalian",
    "gak":"tidak","ga":"tidak","gk":"tidak","nggak":"tidak","ngga":"tidak","ngk":"tidak",
    "kgk":"tidak","tdk":"tidak","engga":"tidak","enggak":"tidak","kga":"tidak",
    "yg":"yang","dgn":"dengan","dg":"dengan","krn":"karena","krna":"karena","utk":"untuk",
    "tp":"tapi","tpi":"tapi","klo":"kalau","klau":"kalau","kl":"kalau","klu":"kalau",
    "spy":"supaya","ato":"atau","bgt":"banget","bgtt":"banget","bngt":"banget","bnr":"benar",
    "bener":"benar","emg":"memang","emang":"memang","mmg":"memang","skrg":"sekarang",
    "skrng":"sekarang","td":"tadi","tdi":"tadi","bsk":"besok","knp":"kenapa",
    "gmn":"bagaimana","gimana":"bagaimana","gmana":"bagaimana","kyk":"seperti",
    "kayak":"seperti","kek":"seperti","udh":"sudah","udah":"sudah","uda":"sudah",
    "dah":"sudah","blm":"belum","blom":"belum","lg":"lagi","lgi":"lagi","msh":"masih",
    "msi":"masih","jd":"jadi","jdi":"jadi","tau":"tahu","taw":"tahu","tw":"tahu",
    "liat":"lihat","ngeliat":"melihat","denger":"dengar","ngomong":"bicara",
    "blng":"bilang","bilg":"bilang","aja":"saja","doang":"saja","doank":"saja",
    "sih":"","deh":"","dong":"","kok":"","lah":"","loh":"","lho":"","yah":"",
    "iy":"iya","iyaa":"iya","iye":"iya","yup":"iya","yoi":"iya","org":"orang",
    "orng":"orang","orgnya":"orangnya","dr":"dari","dri":"dari","kpd":"kepada",
    "thd":"terhadap","ttg":"tentang","abis":"habis","trs":"terus","trus":"terus",
    "tmn":"teman","temen":"teman","tman":"teman","rmh":"rumah","kntr":"kantor",
    "krj":"kerja","ank":"anak","indo":"indonesia","cwo":"cowok","cwe":"cewek",
    "cwok":"cowok","cwek":"cewek","wkwk":"haha","wkwkwk":"haha","wkwkwkwk":"haha",
    "wkwkwkwkwk":"haha","hahaha":"haha","hahahaha":"haha","hehehe":"hehe",
    "hehehehe":"hehe","anjir":"anjing","anjay":"anjing","anjrit":"anjing",
    "anjr":"anjing","anying":"anjing","njir":"anjing","bgsd":"bangsat",
    "bgst":"bangsat","bjir":"anjing","bjg":"bajingan","kntl":"kontol",
    "mmk":"memek","tai":"tahi",
}

assert len(SLANG_DICT) == 149

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\S+\.(?:com|co\.id|net|org|id|tv|me)\b\S*)",
    re.IGNORECASE,
)

MENTION_PATTERN = re.compile(
    r"@\w+"
)

HASHTAG_PATTERN = re.compile(
    r"#(\w+)"
)

ALLOWED_PUNCT = re.compile(
    r"[^\w\s\.,!?\-'\"\u2019]+",
    re.UNICODE,
)

MULTI_SPACE = re.compile(
    r"\s+"
)

def safe_text(value):
    if value is None:
        return ""

    if (
        isinstance(
            value,
            float,
        )
        and pd.isna(
            value
        )
    ):
        return ""

    return str(
        value
    ).strip()

def normalize_slang(text):
    output = []

    for token in text.split():
        prefix = ""
        suffix = ""
        core = token

        while (
            core
            and not core[0].isalnum()
        ):
            prefix += core[0]
            core = core[1:]

        while (
            core
            and not core[-1].isalnum()
        ):
            suffix = (
                core[-1]
                + suffix
            )
            core = core[:-1]

        output.append(
            f"{prefix}"
            f"{SLANG_DICT.get(core.lower(), core)}"
            f"{suffix}"
        )

    return " ".join(
        output
    )

def clean_ocr(text):
    text = safe_text(
        text
    )

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    ).lower()

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = MENTION_PATTERN.sub(
        " ",
        text,
    )

    text = HASHTAG_PATTERN.sub(
        r"\1",
        text,
    )

    text = ALLOWED_PUNCT.sub(
        " ",
        text,
    )

    text = normalize_slang(
        text
    )

    return MULTI_SPACE.sub(
        " ",
        text,
    ).strip()

