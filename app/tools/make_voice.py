"""Pre-record the answer cards with ElevenLabs, once, while online.

The app never calls ElevenLabs: the MP3s ship inside app/audio/ and play offline.
Usage:  python3 tools/make_voice.py --dry-run          # show exactly what will be spoken
        ELEVENLABS_API_KEY=... python3 tools/make_voice.py [--voice VOICE_ID] [--model eleven_v3] [--only sw/rust]
Re-run after any change to app/cards.json (the Swahili must be checked first).
"""
import argparse, datetime, json, os, pathlib, sys, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"

def say(card):
    # What the farmer hears: the name, what to do today, when to call the officer.
    return f"{card['name']}. {card['today']} {card['call']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))
    ap.add_argument("--model", default="eleven_v3")
    ap.add_argument("--langs", default="pt,en")
    ap.add_argument("--only", help="one clip, e.g. sw/rust")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cards = json.loads((ROOT / "cards.json").read_text())
    if a.dry_run:
        for lang in a.langs.split(","):
            for name, card in cards.items():
                if not name.startswith("_"):
                    print(f"{lang}/{name}: {say(card[lang])}\n")
        return
    keyfile = pathlib.Path.home() / ".config" / "elevenlabs" / "api_key"
    key = os.environ.get("ELEVENLABS_API_KEY") or (keyfile.read_text().strip() if keyfile.exists() else None)
    if not key:
        sys.exit(f"set ELEVENLABS_API_KEY or put the key in {keyfile}")
    for lang in a.langs.split(","):
        out = ROOT / "audio" / lang
        out.mkdir(parents=True, exist_ok=True)
        for name, card in cards.items():
            if name.startswith("_") or (a.only and a.only != f"{lang}/{name}"):
                continue
            body = json.dumps({"text": say(card[lang]), "model_id": a.model}).encode()
            req = urllib.request.Request(
                f"https://api.elevenlabs.io/v1/text-to-speech/{a.voice}?output_format=mp3_44100_64",
                data=body, headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
            try:
                mp3 = urllib.request.urlopen(req, timeout=120).read()
            except urllib.error.HTTPError as e:
                sys.exit(f"{lang}/{name}: HTTP {e.code} {e.read()[:300]!r}")
            (out / f"{name}.mp3").write_bytes(mp3)
            print(f"{lang}/{name}.mp3  {len(mp3)//1024} KB")
    # Recorded with what, when: shown in the About screen and cited in the video.
    (ROOT / "audio" / "credits.json").write_text(json.dumps({
        "provider": "ElevenLabs", "model": a.model, "voice_id": a.voice,
        "recorded": datetime.date.today().isoformat(), "cards_status": cards.get("_status", "")}, indent=2))

if __name__ == "__main__":
    main()
