"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const codeTexts = new Map();
  let steps = [];
  let copyTimer;
  const copyTimers = new WeakMap();

  function paintCode(id, value) {
    const element = $(id);
    const source = String(value || "").trim();
    codeTexts.set(id, source);
    element.replaceChildren();
    const token = /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(@model\b|\b(?:from|import|class|def|return|with|as|for|in|if|else|None|True|False)\b)|(\b\d+(?:\.\d+)?\b)/gm;
    let cursor = 0;
    for (const match of source.matchAll(token)) {
      element.appendChild(document.createTextNode(source.slice(cursor, match.index)));
      const span = document.createElement("span");
      span.className = match[1] ? "tok-comment" : match[2] ? "tok-string" : match[3] ? "tok-keyword" : "tok-number";
      span.textContent = match[0];
      element.appendChild(span);
      cursor = match.index + match[0].length;
    }
    element.appendChild(document.createTextNode(source.slice(cursor)));
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* Fall back to a selected text field. */ }
    }
    const focus = document.activeElement;
    const field = document.createElement("textarea");
    field.className = "copy-fallback";
    field.value = text;
    field.setAttribute("readonly", "");
    field.setAttribute("aria-label", "Code to copy");
    document.body.appendChild(field);
    field.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch (_) { copied = false; }
    field.remove();
    if (focus && typeof focus.focus === "function") focus.focus({preventScroll: true});
    return copied;
  }

  document.querySelectorAll("[data-copy]").forEach((button) => {
    const initial = button.textContent;
    button.addEventListener("click", async () => {
      const id = button.dataset.copy;
      const source = codeTexts.get(id);
      if (!source) return;
      const copied = await copyText(source);
      clearTimeout(copyTimer);
      $("copy-status").textContent = copied ? "Code copied to the clipboard." : "Copy is unavailable. Select the code and copy it manually.";
      copyTimer = setTimeout(() => { $("copy-status").textContent = ""; }, 4500);
      if (copied) {
        clearTimeout(copyTimers.get(button));
        button.textContent = "Copied";
        copyTimers.set(button, setTimeout(() => { button.textContent = initial; }, 1800));
      } else {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents($(id));
        if (selection) { selection.removeAllRanges(); selection.addRange(range); }
      }
    });
  });

  function selectStep(index, focus = false, updateHistory = true) {
    if (index < 0 || index >= steps.length) return;
    const step = steps[index];
    paintCode("neural-smallprob", step.smallprob);
    paintCode("neural-pyro", step.pyro);
    $("neural-step-note").textContent = step.note || "Read the steps in order. The downloads contain each complete runnable script.";
    $("neural-comparison").setAttribute("aria-labelledby", `neural-tab-${step.id}`);
    Array.from($("neural-tabs").children).forEach((button, buttonIndex) => {
      const selected = buttonIndex === index;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    const hash = `#${encodeURIComponent(step.id)}`;
    if (updateHistory && window.location.hash !== hash) history.pushState(null, "", hash);
    if (focus) $("neural-tabs").children[index].focus({preventScroll: true});
  }

  function followHash() {
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch (_) { id = ""; }
    const index = steps.findIndex((step) => step.id === id);
    selectStep(index >= 0 ? index : 0, false, false);
  }

  function render(data) {
    if (!Array.isArray(data.steps) || !data.steps.length) throw new Error("No workflow steps in the example.");
    steps = data.steps;
    $("neural-title").textContent = data.title || "A variational autoencoder";
    $("neural-description").textContent = data.description || "";
    $("neural-data-note").textContent = "Synthetic 8 × 8 crossing bars. The Smallprob and Pyro versions train independently with the same architecture and analytic-KL ELBO.";
    $("neural-results-note").textContent = "These results come from executing the supplied example. Training and held-out data are generated separately; reconstruction and generation answer different questions.";
    (data.equations || []).forEach((equation) => {
      const line = document.createElement("p");
      line.textContent = equation;
      $("neural-equations").appendChild(line);
    });
    $("neural-equations").hidden = !data.equations?.length;
    (data.downloads || []).forEach((download) => {
      const link = document.createElement("a");
      link.className = "text-link";
      link.href = download.path;
      link.textContent = `${download.label} ↓`;
      link.setAttribute("download", "");
      $("neural-downloads").appendChild(link);
    });
    steps.forEach((step, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.id = `neural-tab-${step.id}`;
      button.className = "workflow-tab";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", "neural-comparison");
      const number = document.createElement("span");
      number.className = "workflow-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = String(index + 1).padStart(2, "0");
      const label = {data: "Training data", model: "Model & guide", fit: "Fit & evaluate", reconstruct: "Reconstruct", generate: "Generate"}[step.id] || step.title;
      button.append(number, document.createTextNode(label));
      button.addEventListener("click", () => selectStep(index));
      button.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % steps.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + steps.length) % steps.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = steps.length - 1;
        if (next !== undefined) { event.preventDefault(); selectStep(next, true); }
      });
      $("neural-tabs").appendChild(button);
    });
    (data.metrics || []).forEach((metric) => {
      const row = document.createElement("div");
      const label = document.createElement("dt");
      label.textContent = metric.label;
      const value = document.createElement("dd");
      value.textContent = String(metric.value);
      row.append(label, value);
      if (metric.detail) {
        const detail = document.createElement("p");
        detail.className = "metric-detail";
        detail.textContent = metric.detail;
        row.appendChild(detail);
      }
      $("neural-metrics").appendChild(row);
    });
    (data.figures || []).forEach((figure) => {
      const container = document.createElement("figure");
      const fullSize = document.createElement("a");
      fullSize.href = figure.path;
      fullSize.target = "_blank";
      fullSize.rel = "noopener";
      const image = document.createElement("img");
      image.src = figure.path;
      image.alt = figure.alt;
      image.loading = "lazy";
      image.decoding = "async";
      image.addEventListener("error", () => {
        image.hidden = true;
        const error = document.createElement("p");
        error.className = "figure-error";
        error.textContent = "The figure could not load. Use its download link or run the supplied example locally.";
        fullSize.replaceChildren(error);
      }, {once: true});
      fullSize.appendChild(image);
      const caption = document.createElement("figcaption");
      caption.appendChild(document.createTextNode(figure.caption || figure.alt || ""));
      const svg = document.createElement("a");
      svg.href = figure.path;
      svg.textContent = "Open vector figure ↗";
      svg.target = "_blank";
      svg.rel = "noopener";
      caption.appendChild(svg);
      if (figure.png) {
        const png = document.createElement("a");
        png.href = figure.png;
        png.textContent = "300 dpi PNG ↓";
        png.setAttribute("download", "");
        caption.appendChild(png);
      }
      container.append(fullSize, caption);
      $("neural-figures").appendChild(container);
    });
    (data.scope || []).forEach((note) => {
      const item = document.createElement("li");
      item.textContent = note;
      $("neural-scope").appendChild(item);
    });
    (data.sources || []).forEach((source) => {
      const link = document.createElement("a");
      link.href = source.url;
      link.textContent = source.label;
      link.rel = "noopener";
      $("neural-sources").appendChild(link);
    });
    $("neural-load").hidden = true;
    $("neural-content").hidden = false;
    followHash();
  }

  window.addEventListener("hashchange", followHash);
  window.addEventListener("popstate", followHash);
  fetch("assets/vae/example.json?v=0.6.0")
    .then((response) => { if (!response.ok) throw new Error("Example data unavailable."); return response.json(); })
    .then(render)
    .catch(() => {
      $("neural-load").replaceChildren();
      const message = document.createElement("p");
      message.textContent = "The VAE example could not be loaded. Reload the page, or open the complete example on GitHub. A local copy needs to be served over HTTP.";
      const link = document.createElement("a");
      link.href = "https://github.com/nipunbatra/smallprob/tree/main/examples";
      link.textContent = "Open runnable examples →";
      $("neural-load").append(message, link);
    });
})();
