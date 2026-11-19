// Notebook Explainer front end: start a run, stream progress, render documents with
// clickable citations that open the cited code. No build step, no framework.
"use strict";

const $ = (id) => document.getElementById(id);
const TABS = [
  ["manager", "Manager"],
  ["developer", "Developer"],
  ["agent", "AI agent"],
  ["audit", "Audit"],
];
const MAX_CHIPS = 4;
const state = { run: null, claims: {}, tab: "manager", timer: null };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

function parseRef(ref) {
  const m = /^C(\d+):L(\d+)(?:-(\d+))?$/.exec(ref);
  return m ? { cell: +m[1], start: +m[2], end: +(m[3] || m[2]) } : null;
}

// --- starting a run ---------------------------------------------------------

async function init() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    $("models").textContent = `extract ${cfg.model_extract} · write ${cfg.model_write}` +
      (cfg.verify_claims ? ` · verify ${cfg.model_verify}` : "");
    if (cfg.example_path) {
      $("example").addEventListener("click", () => { $("path").value = cfg.example_path; $("path").focus(); });
    } else {
      $("example").classList.add("hidden");
    }
  } catch (e) { /* config is cosmetic */ }
  try { $("path").value = localStorage.getItem("lastPath") || ""; } catch (e) { /* storage blocked */ }
  $("progress-head").addEventListener("click", () => {
    if ($("spinner").classList.contains("stopped")) $("progress").classList.toggle("collapsed");
  });

  $("run-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const path = $("path").value.trim();
    if (!path) { $("path").focus(); return; }
    try { localStorage.setItem("lastPath", path); } catch (e) { /* ignore */ }
    startRun(fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }),
    }));
  });
  $("upload").addEventListener("change", () => {
    const file = $("upload").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    startRun(fetch("/api/runs/upload", { method: "POST", body: form }));
    $("upload").value = "";
  });
}

async function startRun(request) {
  $("error").classList.add("hidden");
  $("result").classList.add("hidden");
  setBusy(true);
  let resp;
  try {
    resp = await request;
  } catch (e) {
    return fail(`Could not reach the server: ${e}`);
  }
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) return fail(body.detail || `Request failed (${resp.status})`);
  follow(body.run_id);
}

function setBusy(busy) {
  document.querySelectorAll("#run-form button, #run-form input").forEach((n) => (n.disabled = busy));
  $("progress").classList.toggle("collapsed", !busy);
  if (busy) {
    $("log").replaceChildren();
    $("progress").classList.remove("hidden");
    $("spinner").classList.remove("stopped");
    $("progress-title").textContent = "Working…";
    const t0 = Date.now();
    clearInterval(state.timer);
    state.timer = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 1000);
  } else {
    clearInterval(state.timer);
    $("spinner").classList.add("stopped");
  }
}

function fail(message) {
  setBusy(false);
  $("progress-title").textContent = "Stopped";
  $("error").textContent = message;
  $("error").classList.remove("hidden");
}

function follow(runId) {
  const source = new EventSource(`/api/runs/${runId}/events`);
  source.onmessage = (ev) => {
    const { message } = JSON.parse(ev.data);
    $("log").append(el("li", { text: message }));
    $("log").scrollTop = $("log").scrollHeight;
  };
  source.addEventListener("done", async () => {
    source.close();
    const run = await (await fetch(`/api/runs/${runId}`)).json();
    setBusy(false);
    $("progress-title").textContent = "Done";
    showResult(run);
  });
  source.addEventListener("error", (ev) => {
    if (ev.data) {  // our own "error" event: the run failed
      source.close();
      fail(`The run failed: ${JSON.parse(ev.data).error}`);
    }
    // otherwise a dropped connection: EventSource reconnects by itself
  });
}

// --- rendering the result ------------------------------------------------------

function showResult(run) {
  state.run = run;
  state.claims = Object.fromEntries(run.claims.claims.map((c) => [c.id, c]));
  $("tabs").replaceChildren(...TABS.map(([key, label]) =>
    el("button", { text: label, "data-tab": key, onclick: () => selectTab(key) })));
  $("result").classList.remove("hidden");
  resetCodePanel();
  selectTab(TABS.some(([k]) => k === state.tab) ? state.tab : "manager");
}

function selectTab(key) {
  state.tab = key;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === key));
  const doc = $("doc");
  if (key === "audit") {
    doc.replaceChildren(renderAudit(state.run));
    $("doc-stats").textContent = "";
    $("download").classList.add("hidden");
    return;
  }
  const d = state.run.documents[key];
  doc.replaceChildren(renderDocument(d));
  const s = d.stats;
  $("doc-stats").textContent = `${s.fact_sentences} cited facts · ${s.guidance_sentences} guidance · ` +
    `${s.claims_cited} claims used · ${s.dropped_sentences} uncited sentences removed`;
  const dl = $("download");
  dl.classList.remove("hidden");
  dl.href = `/api/runs/${state.run.id}/${key}.md`;
  dl.setAttribute("download", `${key}.md`);
}

function renderDocument(d) {
  const frag = document.createDocumentFragment();
  frag.append(el("h1", { text: `${d.title}: ${d.notebook}` }));
  frag.append(el("p", { class: "intro", text:
    "Every factual sentence cites the notebook cell and lines it comes from. Click a citation to see the code; " +
    "sentences in italics are guidance, not facts." }));
  for (const sec of d.sections) {
    frag.append(el("h2", { text: sec.title }));
    if (sec.error) {
      frag.append(el("p", { class: "error", text: `This section could not be generated: ${sec.error}` }));
      continue;
    }
    for (const block of sec.blocks) {
      const items = block.sentences.map(renderSentence);
      if (block.style === "bullets") frag.append(el("ul", {}, items.map((s) => el("li", {}, s))));
      else frag.append(el("p", {}, items.flatMap((s) => [s, " "])));
    }
  }
  return frag;
}

function renderSentence(s) {
  const span = el("span", { class: `sentence${s.kind === "guidance" ? " guidance" : ""}`, text: s.text });
  const wrap = el("span", {}, span);
  const chips = s.refs.map((ref) => el("button", {
    class: "cite", text: ref, title: "Show the code",
    onclick: (ev) => { ev.stopPropagation(); showCode(ref, s, span, ev.currentTarget); },
  }));
  chips.slice(0, MAX_CHIPS).forEach((c) => wrap.append(" ", c));
  if (chips.length > MAX_CHIPS) {
    const rest = chips.slice(MAX_CHIPS);
    const more = el("button", { class: "cite more", text: `+${rest.length}`, title: "Show all citations",
      onclick: () => { more.replaceWith(...rest.flatMap((c) => [" ", c])); } });
    wrap.append(" ", more);
  }
  return wrap;
}

// --- code panel -----------------------------------------------------------------

function resetCodePanel() {
  $("code-panel").replaceChildren(el("div", { class: "code-empty muted" },
    "Click a citation such as ", el("span", { class: "cite static", text: "C26:L320" }),
    " to see the code it refers to."));
}

function showCode(ref, sentence, sentenceEl, chipEl) {
  document.querySelectorAll(".sentence.active, .cite.active").forEach((n) => n.classList.remove("active"));
  sentenceEl.classList.add("active");
  chipEl.classList.add("active");

  const r = parseRef(ref);
  const nb = state.run.notebook;
  const cell = r && nb.cells[r.cell - 1];
  if (!cell) {
    $("code-panel").replaceChildren(el("div", { class: "code-empty", text: `Unknown reference ${ref}` }));
    return;
  }
  const kind = cell.commented_out ? "commented out" : cell.kind;
  const head = el("div", { class: "code-head" },
    el("div", {}, el("span", { class: "title", text: `Cell ${cell.number}` }),
      el("span", { class: "muted", text: ` · ${kind} · lines ${cell.start_line}–${cell.end_line} · citing ${ref}` })));

  const claims = (sentence.claim_ids || []).map((id) => state.claims[id]).filter(Boolean);
  if (claims.length) {
    head.append(el("ul", { class: "claims-list" }, claims.map((c) => {
      const citesThis = c.citations.some((x) => x.ref === ref || overlaps(parseRef(x.ref), r));
      return el("li", { class: citesThis ? "" : "dim", title: citesThis ? "" : "This claim cites other lines" },
        el("span", { class: `badge ${c.verification}`, text: c.verification }),
        el("span", { class: "muted", text: `${c.id} ` }),
        c.statement);
    })));
  }

  // Lines are shown exactly as in the file, so "# MAGIC" cells are not highlighted.
  const lang = !cell.magic && window.hljs && hljs.getLanguage(cell.kind) ? cell.kind : null;
  const pre = el("div", { class: "code" });
  let first = null;
  for (let n = cell.start_line; n <= cell.end_line; n++) {
    const text = nb.lines[n - 1];
    const tx = el("span", { class: "tx" });
    if (window.hljs && lang && !cell.commented_out) {
      tx.innerHTML = hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
    } else {
      tx.textContent = text;
    }
    const line = el("div", { class: `ln${n >= r.start && n <= r.end ? " hl" : ""}` },
      el("span", { class: "no", text: String(n) }), tx);
    if (n === r.start) first = line;
    pre.append(line);
  }
  $("code-panel").replaceChildren(head, pre);
  if (first) first.scrollIntoView({ block: "center" });
}

function overlaps(a, b) {
  return a && b && a.cell === b.cell && a.start <= b.end && b.start <= a.end;
}

// --- audit tab ---------------------------------------------------------------------

function renderAudit(run) {
  const wrap = el("div", { class: "audit" });
  const st = run.claims.stats;
  wrap.append(el("h1", { text: "Audit" }));
  wrap.append(el("p", { class: "intro", text:
    "How the claims behind the documents were checked. Analysis claims (A…) come from static analysis of the code; " +
    "model claims (M…) had every citation checked against the code and were then reviewed by a second model pass." }));
  wrap.append(table([
    ["Claims from static analysis", st.analysis_claims],
    ["Claims from the model (kept)", st.llm_claims],
    ["… verified against the cited code", st.verified],
    ["… corrected by verification", st.corrected],
    ["… not verified", st.unverified],
    ["Citations repaired automatically", st.repaired_citations],
    ["Claims rejected", st.rejected],
    ["LLM calls / cache hits", `${st.llm_calls} / ${st.cache_hits}`],
  ]));

  const docs = Object.values(run.documents);
  wrap.append(el("h2", { text: "Documents" }));
  wrap.append(table([["Document", "Sentences", "Cited facts", "Guidance", "Uncited removed"],
    ...docs.map((d) => [d.title, d.stats.sentences, d.stats.fact_sentences, d.stats.guidance_sentences,
      d.stats.dropped_sentences])], true));

  const corrected = run.claims.claims.filter((c) => c.verification === "corrected");
  if (corrected.length) {
    wrap.append(el("h2", { text: "Claims corrected by verification" }));
    wrap.append(el("ul", {}, corrected.map((c) => el("li", {},
      el("span", { class: "muted", text: `${c.id} ` }), el("span", { class: "old", text: c.original_statement }),
      el("br"), "→ ", c.statement, el("br"), el("span", { class: "muted", text: c.verification_note || "" })))));
  }
  if (run.claims.rejected.length) {
    wrap.append(el("h2", { text: "Rejected claims" }));
    wrap.append(el("ul", {}, run.claims.rejected.map((r) => el("li", {}, r.statement, el("br"),
      el("span", { class: "muted", text: `${r.stage}: ${r.errors.join("; ")}` })))));
  }
  const dropped = docs.flatMap((d) => d.dropped.map((x) => ({ ...x, doc: d.title })));
  if (dropped.length) {
    wrap.append(el("h2", { text: "Sentences removed for lack of a source" }));
    wrap.append(el("ul", {}, dropped.map((x) => el("li", {}, x.text, el("br"),
      el("span", { class: "muted", text: `${x.doc} / ${x.section}: ${x.reason}` })))));
  }
  if (run.claims.errors && run.claims.errors.length) {
    wrap.append(el("h2", { text: "Errors" }));
    wrap.append(el("ul", {}, run.claims.errors.map((e) => el("li", { text: e }))));
  }
  return wrap;
}

function table(rows, header = false) {
  return el("table", {}, rows.map((row, i) => el("tr", {},
    row.map((cell) => el(header && i === 0 ? "th" : "td", { text: String(cell ?? "") })))));
}

init();
