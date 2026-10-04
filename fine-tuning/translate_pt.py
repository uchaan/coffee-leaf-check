"""Offline EN -> pt-BR for the VLM's one-sentence explanation (opus-mt-en-ROMANCE, CTranslate2 int8, 82.3 MB) with the disease names
replaced by "the diagnosis" before translation. Diagnosis cards themselves are a fixed pt-BR list (no generation).
usage: translate_pt.py [MODEL_DIR]   (default translate-en-ptBR-opus-mt-ct2-int8; build it as in README.md)"""
import re, sys, time, ctranslate2, sentencepiece as spm
D = sys.argv[1] if len(sys.argv) > 1 else "translate-en-ptBR-opus-mt-ct2-int8"
# pt-BR diagnosis cards, source of truth for "name" and "looks" in ../app/cards.json (not used by this script)
CARDS = {"A": "Folha sadia — nenhum sinal de doença ou praga.", "B": "Ferrugem-do-cafeeiro — manchas alaranjadas e pó na face inferior da folha.",
         "C": "Cercosporiose (mancha-de-olho-pardo) — manchas marrons redondas com centro claro e halo amarelo.",
         "D": "Mancha-de-phoma — áreas escuras, quase pretas, geralmente na borda ou na ponta da folha.",
         "E": "Bicho-mineiro — manchas marrons secas e irregulares, causadas por larvas dentro da folha.",
         "F": "Não tenho certeza — mostre a folha a um técnico agrícola."}
NAMES = r"(?i)(the diagnosis of )?((coffee )?leaf rust|brown eye spot( \(cercospora\))?|cercospora( leaf spot)?|phoma( leaf spot)?|leaf miners?( damage)?|a healthy leaf)"   # disease names are on the fixed card
tr = ctranslate2.Translator(D, device="cpu", compute_type="int8", inter_threads=1, intra_threads=4)
sp_s = spm.SentencePieceProcessor(model_file=f"{D}/source.spm"); sp_t = spm.SentencePieceProcessor(model_file=f"{D}/target.spm")
def to_pt(en):
    en = re.sub(NAMES, "the diagnosis", en)
    toks = [">>pt_BR<<"] + sp_s.encode(en, out_type=str)
    out = tr.translate_batch([toks], beam_size=4, max_decoding_length=160)[0].hypotheses[0]
    pt = sp_t.decode(out)
    return pt
if __name__ == "__main__":
    for s in ["The leaf shows numerous small, circular spots with yellow-orange fungal growth, scattered across the surface, consistent with coffee leaf rust.",
              "The leaf has brown, dried-out patches, particularly on the right side, consistent with leaf miner damage.",
              "The leaf is green and oval-shaped, with small brown spots visible on the lower surface, consistent with the diagnosis of brown eye spot (Cercospora)."]:
        t = time.time(); print(f"{to_pt(s)}   [{time.time()-t:.2f}s CPU]")
