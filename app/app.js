"use strict";
// Coffee Leaf Check: offline client for llama-server.
// The model only ever returns one letter. Every word the farmer reads or hears comes from
// cards.json (fixed, human-checked text) and pre-recorded audio, so the model cannot invent advice.

// UI strings. Which languages are offered, and in what order, comes from config.json
// ("languages"); the first is the default. The shipped config offers pt and en. The sw strings
// (here, in ABOUT and in cards.json) are an unchecked draft, enabled by adding "sw" to
// config.languages; native speakers have not checked pt or sw yet.
const UI = {
  pt: {
    appName: "Folha de Café",
    hint: "Uma folha só, solta, ocupando a moldura. Depois toque.",
    checking: "Analisando…",
    ok: "Saudável", bad: "Problema encontrado", unsure: "Não tenho certeza",
    today: "Faça hoje", call: "Chame o técnico",
    sure: "de certeza", again: "Analisar outra folha", retake: "Tirar a foto de novo",
    savedNote: "Foto salva. Será enviada ao técnico agrícola quando houver sinal.",
    queueTitle: "Aguardando envio", empty: "Nada aguardando.",
    online: "Com sinal.", offline: "Sem sinal. Serão enviadas depois.",
    sendNow: "Enviar agora", sendStub: "Envio ainda não implementado (demonstração).",
    close: "Fechar", aboutTitle: "Sobre",
    camErr: "Câmera indisponível. Use o botão de foto à esquerda.",
    bestGuess: "palpite",
    noVoice: "Este celular não tem voz em português sem internet.",
    likely: "provável", veryLikely: "muito provável",
    notLeaf: "Isto não parece uma folha. Fotografe uma folha só, ocupando a moldura.",
  },
  sw: {
    appName: "Jani la Kahawa",
    hint: "Jani moja lililochumwa, lijaze fremu. Kisha bonyeza.",
    checking: "Inakagua…",
    ok: "Afya njema", bad: "Tatizo limeonekana", unsure: "Sina uhakika",
    today: "Fanya leo", call: "Mpigie afisa ugani",
    sure: "uhakika", again: "Kagua jani lingine", retake: "Piga picha tena",
    savedNote: "Picha imehifadhiwa. Itatumwa kwa afisa ugani mtandao ukipatikana.",
    queueTitle: "Zinasubiri kutumwa", empty: "Hakuna picha zinazosubiri.",
    online: "Mtandao upo.", offline: "Hakuna mtandao. Zitatumwa baadaye.",
    sendNow: "Tuma sasa", sendStub: "Kutuma bado hakujajengwa (mfano tu).",
    close: "Funga", aboutTitle: "Kuhusu",
    camErr: "Kamera haipatikani. Tumia kitufe cha picha upande wa kushoto.",
    bestGuess: "makisio",
    noVoice: "Simu hii haina sauti ya Kiswahili bila mtandao.",
    likely: "inawezekana", veryLikely: "inawezekana sana",
    notLeaf: "Hii haionekani kama jani. Piga picha ya jani moja, lijaze fremu.",
  },
  en: {
    appName: "Coffee Leaf Check",
    hint: "One loose leaf, filling the frame. Then tap.",
    checking: "Checking…",
    ok: "Healthy", bad: "Problem found", unsure: "Not sure",
    today: "Do today", call: "Call the extension officer",
    sure: "sure", again: "Check another leaf", retake: "Take the photo again",
    savedNote: "Photo saved. It will be sent to the extension officer when there is signal.",
    queueTitle: "Waiting to send", empty: "Nothing waiting.",
    online: "Online.", offline: "No signal. They will send later.",
    sendNow: "Send now", sendStub: "Sending is not built yet (stub).",
    close: "Close", aboutTitle: "About",
    camErr: "Camera not available. Use the photo button on the left.",
    bestGuess: "best guess",
    noVoice: "This phone has no offline voice for English.",
    likely: "likely", veryLikely: "very likely",
    notLeaf: "This does not look like a leaf. Photograph one leaf, filling the frame.",
  },
};

const ABOUT = {
  pt: `<p><b>O que faz.</b> Analisa uma folha de café arábica e identifica cinco situações: saudável, ferrugem, cercosporiose, mancha de Phoma e bicho-mineiro. Funciona neste celular, sem internet.</p>
<p><b>O que não faz.</b> Não recomenda produtos químicos nem doses. Quando não tem certeza, diz isso e guarda a foto para o técnico agrícola. Quem decide é uma pessoa.</p>
<p><b>Limites.</b> Testado com fotos do BRACOL (Brasil: uma folha solta, fundo claro) e do JMuBEN (Quênia: close-ups tirados no pé). Não foi testado com fotos com solo, galhos ou outras folhas no quadro, outras variedades, outras doenças nem fotos da planta inteira.</p>
<p><b>Precisão.</b> 90,8% no conjunto de teste do BRACOL (1.266 fotos). Em fotos de outro tipo, a precisão cai bastante.</p>
<p class="small">Dados: BRACOL (Krohling, Esgario, Ventura) e JMuBEN (Jepkoech et al.), Mendeley Data, CC BY 4.0. Modelo base: Qwen3.5-2B, Apache 2.0, quantizado e ajustado pela equipe. Execução: llama.cpp (MIT). Voz: ElevenLabs, gravada antes e tocada sem internet.</p>`,
  en: `<p><b>What it does.</b> Checks one Arabica coffee leaf for five conditions: healthy, leaf rust, brown eye spot, Phoma, leaf miner. Runs on this phone with no internet.</p>
<p><b>What it does not do.</b> It does not recommend chemicals or doses. When it is not sure, it says so and saves the photo for the extension officer. A person makes the final call.</p>
<p><b>Limits.</b> Tested on BRACOL (Brazil: one picked leaf on a plain background) and JMuBEN (Kenya: close-ups taken on the plant) photos. Not tested on photos with soil, branches or other leaves in the frame, other varieties, other diseases, or whole trees.</p>
<p><b>Accuracy.</b> 90.8% on the BRACOL test set (1,266 photos). On other kinds of photos it drops a lot.</p>
<p class="small">Data: BRACOL (Krohling, Esgario, Ventura) and JMuBEN (Jepkoech et al.), Mendeley Data, CC BY 4.0. Base model: Qwen3.5-2B, Apache 2.0, quantized and fine-tuned by the team. Runtime: llama.cpp (MIT). Voice: ElevenLabs, recorded in advance and played offline.</p>`,
  // sw: draft, not offered by default; Limits/Accuracy not yet updated to match pt and en.
  sw: `<p><b>Kinachofanya.</b> Hukagua jani moja la kahawa ya Arabika kwa hali tano: lenye afya, kutu, madoa ya jicho kahawia, Phoma, mchimba jani. Hufanya kazi kwenye simu hii bila mtandao.</p>
<p><b>Kisichofanya.</b> Hakipendekezi dawa wala vipimo. Kisipokuwa na uhakika, husema hivyo na kuhifadhi picha kwa ajili ya afisa ugani. Uamuzi wa mwisho ni wa mtu.</p>
<p><b>Mipaka.</b> Kimejaribiwa kwa picha za BRACOL kutoka Brazili: jani moja moja, mandharinyuma safi. Hakijajaribiwa kwa aina nyingine, nchi nyingine, magonjwa mengine, wala picha za mti mzima.</p>
<p class="small">Hifadhidata ya BRACOL (Krohling, Esgario, Ventura; Mendeley Data, CC BY 4.0). Modeli: Qwen3.5 (Alibaba Qwen). Programu: llama.cpp. Sauti: ElevenLabs, imerekodiwa mapema na huchezwa bila mtandao.</p>`,
};

const $ = (s) => document.querySelector(s);
const safeGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const safeSet = (k, v) => { try { localStorage.setItem(k, v); } catch {} };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let CFG, PROMPT, CARDS, last = null, sheetMode = null, busy = false;
const AUDIO = new Set(); // "sw/rust" etc. that exist on disk
let lang = safeGet("lang");
const t = (k) => UI[lang][k] ?? k;

// ---------- boot ----------
async function boot() {
  try {
    CFG = await (await fetch("config.json", { cache: "no-store" })).json();
    if (new URLSearchParams(location.search).has("mock")) CFG.mock = true;
    PROMPT = parsePrompt(await (await fetch(CFG.prompt_file, { cache: "no-store" })).text());
    CARDS = await (await fetch("cards.json", { cache: "no-store" })).json();
  } catch (e) {
    return showErr("Startup failed: " + e.message);
  }
  const langs = (CFG.languages || ["pt", "en"]).filter((l) => UI[l]);
  if (!langs.includes(lang)) lang = langs[0];
  $("#langchip").innerHTML = langs.map((l) => `<button data-lang="${l}">${l.toUpperCase()}</button>`).join("");
  await findAudio(langs);
  document.querySelectorAll("#langchip button").forEach((b) =>
    b.addEventListener("click", () => { lang = b.dataset.lang; safeSet("lang", lang); applyLang(); }));
  $("#shutter").addEventListener("click", onShutter);
  $("#file").addEventListener("change", (ev) => { const f = ev.target.files[0]; ev.target.value = ""; if (f) check(f); });
  $("#btn-queue").addEventListener("click", () => openPanel("queue"));
  $("#btn-about").addEventListener("click", () => openPanel("about"));
  $("#sheet").addEventListener("click", (e) => {
    const a = e.target.closest("[data-act]");
    if (!a) return;
    if (a.dataset.act === "again") closeSheet();
    if (a.dataset.act === "speak") speak();
    if (a.dataset.act === "send") $("#qmsg").textContent = t("sendStub");
  });
  addEventListener("online", () => sheetMode === "queue" && openPanel("queue"));
  addEventListener("offline", () => sheetMode === "queue" && openPanel("queue"));
  applyLang();
  startCamera();
}

// Package format (protocol/prompt_v1.txt, copied as prompt.txt):
// "### system", "### user", "### grammar" sections, used word for word.
// A plain "SYSTEM: ... USER: ..." file is also accepted (no grammar).
function parsePrompt(txt) {
  txt = txt.replace(/\r\n/g, "\n");
  if (/^### system/m.test(txt)) {
    const sec = {};
    for (const part of txt.split(/^### /m).slice(1)) {
      const nl = part.indexOf("\n");
      sec[part.slice(0, nl).trim().toLowerCase()] = part.slice(nl + 1).trim();
    }
    if (!sec.system || !sec.user) throw new Error("prompt.txt: missing ### system or ### user");
    return { system: sec.system, user: sec.user, grammar: sec.grammar || null };
  }
  const m = txt.match(/^SYSTEM:[ \t]*([\s\S]*?)\nUSER[^\n:]*:[ \t]*\n?([\s\S]*)$/);
  if (!m) throw new Error("prompt.txt: expected ### system / ### user sections");
  return { system: m[1].trim(), user: m[2].trim(), grammar: null };
}

async function findAudio(langs) {
  const keys = Object.keys(CARDS).filter((k) => !k.startsWith("_"));
  await Promise.all(langs.flatMap((l) => keys.map(async (k) => {
    try { if ((await fetch(`audio/${l}/${k}.mp3`, { method: "HEAD" })).ok) AUDIO.add(`${l}/${k}`); } catch {}
  })));
}

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("#langchip button").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
  $(".brand span").textContent = t("appName");
  $("#hint").textContent = busy ? t("checking") : t("hint");
  if ($("#camerr").style.display === "block") $("#camerr").textContent = t("camErr");
  if (sheetMode === "result" && last) renderResult(last, false);
  else if (sheetMode) openPanel(sheetMode);
  refreshCount();
}

function showErr(msg) {
  const e = $("#err");
  e.textContent = msg;
  e.style.display = "block";
  clearTimeout(showErr.tm);
  showErr.tm = setTimeout(() => { e.style.display = "none"; }, 6000);
}

// ---------- camera ----------
let stream = null;
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
    });
    $("#video").srcObject = stream;
    $("#camerr").style.display = "none";
  } catch (e) {
    console.warn("camera", e);
    $("#camerr").textContent = t("camErr");
    $("#camerr").style.display = "block";
  }
}

function onShutter() {
  if (busy) return;
  if (sheetMode) return closeSheet();
  const v = $("#video");
  if (!stream || !v.videoWidth) return $("#file").click();
  check(v);
}

// ---------- the check ----------
async function check(source) {
  busy = true;
  document.body.classList.add("busy");
  $("#hint").textContent = t("checking");
  stopVoice();
  try {
    // Freeze what was checked on screen.
    const bmp = source instanceof HTMLVideoElement
      ? await createImageBitmap(source)
      : await createImageBitmap(source, { imageOrientation: "from-image" });
    $("#still").src = await toBlobURL(bmp, Math.max(bmp.width, bmp.height), 0.85);
    $("#still").style.display = "block";
    $("#video").pause();

    const img = await resize(bmp);
    const t0 = performance.now();
    // Leaf check first (see config.leaf_gate); only a leaf goes to the package diagnosis.
    const isLeaf = await leafGate(img.dataUrl);
    const probs = isLeaf ? await classify(img.dataUrl) : null;
    const ms = Math.round(performance.now() - t0);
    const d = isLeaf ? decide(probs) : { top: null, p: 0, unsure: true, key: "unsure", notLeaf: true };
    last = { ...d, probs, ms, w: img.w, h: img.h };
    if (d.unsure && !d.notLeaf) await queueAdd({ time: Date.now(), blob: img.blob, top: d.top, p: d.p, probs, model: CFG.model_label });
    console.log("[leaf]", JSON.stringify({ ms, top: d.top, p: d.p, unsure: d.unsure, probs }));
    renderResult(last, true);
    refreshCount();
  } catch (e) {
    showErr(e.message);
    resumeCamera();
  } finally {
    busy = false;
    document.body.classList.remove("busy");
    $("#hint").textContent = t("hint");
  }
}

async function toBlobURL(bmp, side, q) {
  const s = Math.min(1, side / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return URL.createObjectURL(await new Promise((r) => c.toBlob(r, "image/jpeg", q)));
}

// Protocol: longest side -> 512 px, JPEG quality 0.9.
async function resize(bmp) {
  const s = CFG.image.longest_side / Math.max(bmp.width, bmp.height);
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", CFG.image.jpeg_quality));
  const dataUrl = await new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  return { blob, dataUrl, w, h };
}

function grammarFor(letters) {
  return `root ::= [${letters[0]}-${letters[letters.length - 1]}]`;
}

// The package diagnosis request, exactly as specified.
async function classify(dataUrl) {
  const letters = CFG.letters;
  if (CFG.mock) return mockProbs(letters);
  return letterProbs(await ask(dataUrl, PROMPT.system, PROMPT.user, PROMPT.grammar || grammarFor(letters)), letters);
}

async function leafGate(dataUrl) {
  const g = CFG.leaf_gate;
  if (!g || !g.enabled || CFG.mock) return true;
  // Package system text, adapter as loaded: the fine-tuned model still answers "is this a leaf?" well.
  const lora = g.lora_scale == null ? undefined : [{ id: 0, scale: g.lora_scale }];
  const p = letterProbs(await ask(dataUrl, g.system || PROMPT.system, g.user, g.grammar, lora), "AB");
  console.log("[gate]", JSON.stringify(p));
  return p.A > p.B;
}

async function ask(dataUrl, system, user, grammar, lora) {
  const body = {
    messages: [
      { role: "system", content: system },
      { role: "user", content: [
        { type: "image_url", image_url: { url: dataUrl } },
        { type: "text", text: user },
      ] },
    ],
    grammar,
    chat_template_kwargs: CFG.chat_template_kwargs,
    ...CFG.sampling,
    ...(lora ? { lora } : {}),
  };
  const r = await fetch(CFG.server + "/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Model server error ${r.status}: ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

// Read the per-letter probabilities of the first generated token.
// With post_sampling_probs llama-server returns {token, prob} in top_probs;
// otherwise {token, logprob} in top_logprobs. Accept either.
function letterProbs(resp, letters) {
  const first = resp?.choices?.[0]?.logprobs?.content?.[0];
  if (!first) throw new Error("No token probabilities in the model response");
  const list = first.top_probs || first.top_logprobs || [];
  const p = Object.fromEntries([...letters].map((L) => [L, 0]));
  for (const e of list) {
    const tok = String(e.token ?? ""); // exact token: " A" or "a" are not the grammar's letters
    if (tok in p) p[tok] += e.prob ?? Math.exp(e.logprob);
  }
  return p;
}

// Answer = letter with the highest returned probability (the sampled token is ignored).
// Not sure = top letter is the unsure option, or its probability is below tau.
function decide(probs) {
  let top = null, p = -1;
  for (const [L, v] of Object.entries(probs)) if (v > p) { top = L; p = v; }
  const unsure = top === CFG.unsure_letter || p < CFG.tau;
  return { top, p, unsure, key: unsure ? "unsure" : CFG.classes[top] };
}

function mockProbs(letters) {
  const raw = [...letters].map(() => Math.random() ** 3);
  const sum = raw.reduce((a, b) => a + b, 0);
  return new Promise((r) => setTimeout(() => r(Object.fromEntries([...letters].map((L, i) => [L, raw[i] / sum]))), 900));
}

// ---------- sheet ----------
function openSheet(html, mode) {
  sheetMode = mode;
  $("#sheet-body").innerHTML = html;
  $("#sheet").scrollTop = 0;
  $("#sheet").classList.add("open");
}
function closeSheet() {
  $("#sheet").classList.remove("open");
  sheetMode = null;
  stopVoice();
  resumeCamera();
}
function resumeCamera() {
  $("#still").style.display = "none";
  $("#video").play().catch(() => {});
}

function renderResult(r, autoplay) {
  const c = CARDS[r.key][lang];
  const kind = r.unsure ? "unsure" : r.key === "healthy" ? "ok" : "bad";
  const pct = Math.round(r.p * 100);
  const word = r.p >= (CFG.very_likely ?? 0.8) ? t("veryLikely") : t("likely");
  const enName = lang !== "en" && CARDS[r.key].en ? CARDS[r.key].en.name : "";
  const hasVoice = true; // recorded clip, else the phone's own voice (see speak)
  openSheet(
    `${r.unsure ? "" : `<span class="status s-${kind}">${t(kind)}</span>`}
     <div class="title"><h2>${esc(c.name)}</h2>${hasVoice ? `<button class="speak" data-act="speak" aria-label="play">${SPEAKER}</button>` : ""}</div>
     ${enName ? `<p class="en-name">${esc(enName)}</p>` : ""}
     <p class="looks">${r.notLeaf ? t("notLeaf") : esc(c.looks)}</p>
     ${r.unsure ? "" : `<div class="conf s-${kind}" style="background:none"><div class="bar"><i style="width:${pct}%"></i></div><span>${word}</span></div>`}
     <div class="block"><h3>${t("today")}</h3><p>${esc(c.today)}</p></div>
     ${r.notLeaf ? "" : `<div class="block"><h3>${t("call")}</h3><p>${esc(c.call)}</p></div>`}
     ${r.unsure && !r.notLeaf ? `<p class="saved">${t("savedNote")}</p>` : ""}
     <p class="src">${esc(c.source)}</p>
     <button class="cta" data-act="again">${r.unsure ? t("retake") : t("again")}</button>
     <p class="meta">${CFG.mock ? "MOCK · " : ""}${esc(CFG.model_label)} · ${r.ms} ms · offline</p>`,
    "result");
  if (autoplay && hasVoice) speak();
}

function openPanel(which) {
  if (which === "about") {
    openSheet(`<div class="panel"><h2>${t("aboutTitle")}</h2>${ABOUT[lang]}<button class="cta" data-act="again">${t("close")}</button></div>`, "about");
    return;
  }
  queueAll().then((items) => {
    const rows = items.length
      ? items.reverse().map((it) => `<div class="q"><img src="${URL.createObjectURL(it.blob)}" alt="">
          <div><div>${esc(new Date(it.time).toLocaleString())}</div>
          <div class="small">${t("bestGuess")}: ${esc(it.top)} (${Math.round(it.p * 100)}%)</div></div></div>`).join("")
      : `<p class="small">${t("empty")}</p>`;
    openSheet(`<div class="panel"><h2>${t("queueTitle")}</h2>
      <p class="small">${navigator.onLine ? t("online") : t("offline")}</p>${rows}
      <button class="cta" data-act="send">${t("sendNow")}</button><p class="small" id="qmsg"></p>
      <button class="cta" style="background:none;color:var(--ink);border:1px solid var(--line)" data-act="again">${t("close")}</button></div>`, "queue");
  });
}

// ---------- voice, all offline ----------
// 1. ElevenLabs clip recorded in advance (audio/<lang>/<card>.mp3), best quality.
// 2. Else the phone's own text-to-speech, if it has a voice for the language installed.
// 3. Else say so; never read Swahili text with an English voice.
const SPEAKER = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
const TTS_LANG = { pt: "pt-BR", sw: "sw-KE", en: "en-US" };
const setPlaying = (on) => document.querySelector(".speak")?.classList.toggle("playing", on);

function speak() {
  if (!last) return;
  if (!AUDIO.has(`${lang}/${last.key}`)) return speakDevice(CARDS[last.key][lang]);
  const a = $("#voice");
  a.src = `audio/${lang}/${last.key}.mp3`;
  a.onplay = () => document.querySelector(".speak")?.classList.add("playing");
  a.onended = a.onpause = () => document.querySelector(".speak")?.classList.remove("playing");
  a.play().catch(() => {});
}
function speakDevice(card) {
  if (!("speechSynthesis" in window)) return showErr(t("noVoice"));
  speechSynthesis.cancel();
  // Same words as the recorded clips (tools/make_voice.py).
  const u = new SpeechSynthesisUtterance(`${card.name}. ${card.today} ${card.call}`);
  u.lang = TTS_LANG[lang];
  u.onstart = () => setPlaying(true);
  u.onend = () => setPlaying(false);
  u.onerror = (e) => { setPlaying(false); if (e.error !== "interrupted" && e.error !== "canceled") showErr(t("noVoice")); };
  speechSynthesis.speak(u);
}
function stopVoice() { $("#voice").pause(); if ("speechSynthesis" in window) speechSynthesis.cancel(); }

// ---------- send queue (IndexedDB; sending is a stub) ----------
function db() {
  return new Promise((res, rej) => {
    const q = indexedDB.open("leafcheck", 1);
    q.onupgradeneeded = () => q.result.createObjectStore("queue", { keyPath: "id", autoIncrement: true });
    q.onsuccess = () => res(q.result);
    q.onerror = () => rej(q.error);
  });
}
async function queueTx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction("queue", mode);
    const out = fn(tx.objectStore("queue"));
    tx.oncomplete = () => res(out.result);
    tx.onerror = () => rej(tx.error);
  });
}
async function queueAdd(item) {
  try { await queueTx("readwrite", (s) => s.add(item)); } catch (e) { console.warn("queue add failed", e); }
}
async function queueAll() {
  try { return await queueTx("readonly", (s) => s.getAll()); } catch { return []; }
}
async function refreshCount() {
  const n = (await queueAll()).length;
  $("#qcount").hidden = !n;
  $("#qcount").textContent = n;
}

boot();
