"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const caseData = window.SMALLPROB_CASES;
  const codeTexts = new Map();
  const buttonTimers = new WeakMap();
  let activeIndex = 0;
  let copyTimer;

  function showCopyStatus(message) {
    clearTimeout(copyTimer);
    $("copy-status").textContent = message;
    copyTimer = setTimeout(() => { $("copy-status").textContent = ""; }, 4500);
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_) { /* Try selection-based copying if browser permission is denied. */ }
    }
    const priorFocus = document.activeElement;
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
    if (priorFocus && typeof priorFocus.focus === "function") priorFocus.focus({ preventScroll: true });
    return copied;
  }

  document.querySelectorAll("[data-copy]").forEach((button) => {
    const originalLabel = button.textContent;
    button.addEventListener("click", async () => {
      const codeId = button.dataset.copy;
      const text = codeTexts.get(codeId) ?? $(codeId)?.textContent;
      if (!text) {
        showCopyStatus("No code is available to copy yet.");
        return;
      }
      const copied = await copyText(text);
      if (copied) {
        clearTimeout(buttonTimers.get(button));
        button.textContent = "Copied";
        showCopyStatus("Code copied to the clipboard.");
        buttonTimers.set(button, setTimeout(() => { button.textContent = originalLabel; }, 1800));
      } else {
        showCopyStatus("Copy is unavailable in this browser. Select the code and copy it manually.");
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents($(codeId));
        if (selection) {
          selection.removeAllRanges();
          selection.addRange(range);
        }
      }
    });
  });

  function paintCode(id, value) {
    const element = $(id);
    const source = String(value || "").trim();
    codeTexts.set(id, source);
    element.replaceChildren();
    const token = /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(@model\b|\b(?:from|import|def|return|with|as|for|in|if|else|None|True|False)\b)|(\b\d+(?:\.\d+)?\b)/gm;
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

  function assetFailed(image, message) {
    const previous = image.parentElement.querySelector(".figure-error");
    if (previous) previous.remove();
    const error = document.createElement("p");
    error.className = "figure-error";
    error.setAttribute("role", "status");
    error.textContent = message;
    image.insertAdjacentElement("afterend", error);
    image.hidden = true;
  }

  function setImage(image, path, alt) {
    image.parentElement.querySelectorAll(".figure-error").forEach((node) => node.remove());
    image.hidden = false;
    image.alt = alt;
    image.src = path;
  }

  $("case-diagram").addEventListener("error", () => assetFailed($("case-diagram"), "The diagram could not be loaded. Use the full-size link or run the downloaded example to draw it locally."));
  $("case-plot").addEventListener("error", () => assetFailed($("case-plot"), "The plot could not be loaded. The written result and runnable example are available above."));

  if (!Array.isArray(caseData) || caseData.length === 0) {
    $("load-state").replaceChildren();
    const heading = document.createElement("p");
    heading.textContent = "The example data could not be loaded.";
    const detail = document.createElement("p");
    detail.textContent = "Reload this page, or open one of the complete notebooks below. If you are running a local copy, keep the assets folder beside index.html.";
    $("load-state").append(heading, detail);
    $("case-count").textContent = "00";
    return;
  }

  const cases = caseData.filter((item) => item && typeof item.id === "string" && typeof item.title === "string");
  if (!cases.length) {
    $("load-state").textContent = "No model examples are available in this copy. The complete notebooks are linked below.";
    $("case-count").textContent = "00";
    return;
  }
  $("case-count").textContent = String(cases.length).padStart(2, "0");

  function selectCase(index, focusTab = false) {
    if (index < 0 || index >= cases.length) return;
    const hash = `#${encodeURIComponent(cases[index].id)}`;
    if (window.location.hash !== hash) history.pushState(null, "", hash);
    renderCase(index);
    if (focusTab) $("case-tabs").children[index].focus({ preventScroll: true });
  }

  cases.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-tab";
    button.id = `tab-${item.id}`;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", "case-panel");
    button.setAttribute("aria-selected", "false");
    button.tabIndex = -1;
    const number = document.createElement("span");
    number.className = "case-number";
    number.setAttribute("aria-hidden", "true");
    number.textContent = String(index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = item.title;
    button.append(number, label);
    button.addEventListener("click", () => selectCase(index));
    button.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % cases.length;
      else if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (index - 1 + cases.length) % cases.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = cases.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        selectCase(next, true);
      }
    });
    $("case-tabs").appendChild(button);
  });

  function renderCase(index) {
    activeIndex = index;
    const item = cases[index];
    $("navigation-status").hidden = true;
    $("case-eyebrow").textContent = item.eyebrow || `Example ${String(index + 1).padStart(2, "0")}`;
    $("case-title").textContent = item.title;
    $("case-description").textContent = item.description || "";
    $("case-method").textContent = item.method || "Inference example";
    $("case-method").dataset.approximate = String(/variational|\bVI\b|approximate/i.test(item.method || ""));
    $("case-panel").setAttribute("aria-labelledby", `tab-${item.id}`);
    paintCode("smallprob-code", item.smallprob);
    paintCode("pyro-code", item.pyro);
    paintCode("run-code", item.run);
    $("case-result").textContent = item.result || "Run the example to inspect its posterior.";
    $("example-download").href = item.download || "https://github.com/nipunbatra/smallprob";
    $("example-download").setAttribute("aria-label", `Download the runnable ${item.title} example`);
    $("diagram-full").href = item.diagram || "#";
    $("diagram-link").href = item.diagram || "#";
    setImage($("case-diagram"), item.diagram || "", `${item.title}: generated graphical model and mathematical equations`);
    $("plot-figure").hidden = !item.plot;
    $("plot-full").hidden = !item.plot;
    if (item.plot) {
      $("plot-full").href = item.plot;
      setImage($("case-plot"), item.plot, `${item.title}: inference results. ${item.result || ""}`);
    }
    $("case-notes").replaceChildren();
    (item.notes || []).forEach((note) => {
      const li = document.createElement("li");
      li.textContent = note;
      $("case-notes").appendChild(li);
    });
    $("case-sources").replaceChildren();
    (item.sources || []).forEach((source) => {
      const link = document.createElement("a");
      link.textContent = source.label;
      link.href = source.url;
      if (/^https?:\/\//.test(source.url)) link.rel = "noopener";
      $("case-sources").appendChild(link);
    });
    Array.from($("case-tabs").children).forEach((button, buttonIndex) => {
      const selected = buttonIndex === index;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("previous-case").disabled = index === 0;
    $("next-case").disabled = index === cases.length - 1;
    $("load-state").hidden = true;
    $("case-panel").hidden = false;
    document.title = `${item.title} · Smallprob`;
  }

  function followHash() {
    let requested;
    try { requested = decodeURIComponent(window.location.hash.slice(1)); } catch (_) { requested = ""; }
    if (requested === "get-started" || requested === "examples") return;
    const index = cases.findIndex((item) => item.id === requested);
    renderCase(index >= 0 ? index : 0);
    if (requested && index < 0) {
      $("navigation-status").textContent = `That example link was not found. Showing ${cases[0].title}.`;
      $("navigation-status").hidden = false;
    }
  }

  $("previous-case").addEventListener("click", () => {
    selectCase(activeIndex - 1);
    $("examples").scrollIntoView({ block: "start" });
    $("case-panel").focus({ preventScroll: true });
  });
  $("next-case").addEventListener("click", () => {
    selectCase(activeIndex + 1);
    $("examples").scrollIntoView({ block: "start" });
    $("case-panel").focus({ preventScroll: true });
  });
  window.addEventListener("hashchange", followHash);
  window.addEventListener("popstate", followHash);
  renderCase(0);
  followHash();
})();
