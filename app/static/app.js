const form = document.querySelector("#qa-form");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status");
const requestInfo = document.querySelector("#request-info");
const resultsSection = document.querySelector("#results-section");
const results = document.querySelector("#results");

const coverageLabels = {
  found: ["Found", "The model reports support for every part of this question."],
  partial: ["Partial", "The model reports support for only part of this question. Review the gaps and cited evidence."],
  not_found: ["Not found", "The model found no answer in the retrieved evidence. This does not mean the answer is no."],
  unknown: ["Status unavailable", "Evidence coverage was not reported. Review the answer and citations."],
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderResults(items) {
  results.replaceChildren();
  for (const [index, item] of items.entries()) {
    const card = document.createElement("article");
    card.className = "panel answer";
    card.append(element("h3", `${index + 1}. ${item.question}`));
    const coverage = Object.hasOwn(coverageLabels, item.status) ? item.status : "unknown";
    const [label, explanation] = coverageLabels[coverage];
    card.append(element("span", label, `support-status support-${coverage}`));
    card.append(element("p", explanation, "note"));
    card.append(element("p", item.answer, "answer-text"));
    if (item.citations.length > 0) {
      const evidence = document.createElement("details");
      evidence.open = true;
      evidence.append(element("summary", coverage === "partial" ? "Source evidence — partial support" : "Source evidence"));
      for (const citation of item.citations) {
        evidence.append(element("p", citation.page === null ? "JSON source" : `Page ${citation.page}`, "source-label"));
        evidence.append(element("blockquote", citation.excerpt));
      }
      card.append(evidence);
    } else {
      card.append(element("p", "No supporting citation returned.", "note"));
    }
    results.append(card);
  }
  resultsSection.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitButton.disabled || !form.reportValidity()) return;
  submitButton.disabled = true;
  form.setAttribute("aria-busy", "true");
  resultsSection.hidden = true;
  results.replaceChildren();
  requestInfo.textContent = "";
  statusMessage.className = "";
  statusMessage.textContent = "Processing document and questions…";
  const started = performance.now();
  const timer = window.setInterval(() => {
    statusMessage.textContent = `Processing… ${Math.floor((performance.now() - started) / 1000)}s elapsed.`;
  }, 1000);

  try {
    const response = await fetch("/qa", { method: "POST", body: new FormData(form) });
    const requestId = response.headers.get("X-Request-ID");
    requestInfo.textContent = requestId ? `Request ID: ${requestId}` : "";
    const payload = await response.json();
    if (!response.ok) {
      const message = payload.error?.message ?? "The request could not be completed.";
      throw new Error(`${message} (HTTP ${response.status})`);
    }
    if (!Array.isArray(payload.results)) throw new Error("The server returned an unexpected response.");
    renderResults(payload.results);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    statusMessage.textContent = `Completed ${payload.results.length} questions in ${seconds}s.`;
  } catch (error) {
    statusMessage.className = "error";
    statusMessage.textContent = error instanceof Error ? error.message : "Unable to reach the server. Please try again.";
  } finally {
    window.clearInterval(timer);
    submitButton.disabled = false;
    form.setAttribute("aria-busy", "false");
  }
});
