(() => {
  "use strict";

  // Observatory G1.3.3.1 - geographic focus interaction correctness pass.

  const shell = document.getElementById("observatoryShell");
  const stage = document.getElementById("observatoryStage");
  const mapStage = document.getElementById("observatoryMapStage");
  const map = document.getElementById("observatoryNationalMap");
  const mapPaths = document.getElementById("observatoryMapPaths");
  const mapLoading = document.getElementById("observatoryMapLoading");
  const jurisdictionPicker = document.getElementById("observatoryJurisdictionPicker");
  const mapSignal = document.getElementById("observatoryMapSignal");
  const contextPanel = document.querySelector(".observatory-context");
  const contextTitle = document.getElementById("observatoryContextTitle");
  const contextCopy = document.getElementById("observatoryContextCopy");
  const contextCode = document.getElementById("observatoryContextCode");
  const contextStatus = document.getElementById("observatoryContextStatus");
  const lensTitle = document.getElementById("observatoryLensTitle");
  const lensCopy = document.getElementById("observatoryLensCopy");
  const phrase = document.getElementById("observatoryPhrase");
  const pathways = Array.from(document.querySelectorAll("[data-observatory-pathway]"));
  const projectLink = document.getElementById("observatoryProjectLink");
  function updateProjectPrefill() {
    if (!projectLink) return;
    // Geography is a UI prefill, never project, registry or canonical authority.
    const state = selectedJurisdiction && selectedJurisdiction.abbr;
    projectLink.setAttribute("href", state && /^[A-Z]{2}$/.test(state)
      ? `/projects?state=${encodeURIComponent(state)}` : "/projects");
  }
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const phrases = [
    "Preserve the source. Understand the result.",
    "Scale should create wonder. Detail should create understanding.",
    "Follow the evidence.",
    "From jurisdiction to precinct.",
    "Local data matters."
  ];

  let phraseIndex = 0;
  const CONUS_REFLECT_Y = 483;

  let selectedPath = null;
  let selectedFeature = null;
  let selectedJurisdiction = null;
  let signalTrace = null;
  let jurisdictionPickerEntries = new Map();
  let geographySequence = 0;
  let geographyActive = null;
  // Insert recovery controls alongside the existing selector, without replacing
  // the homepage template (which the donation stream is editing in parallel).
  const geographyActions = document.createElement('div');
  geographyActions.className = 'observatory-geo-actions';
  const clearButton = document.createElement('button');
  clearButton.type = 'button';
  clearButton.textContent = 'Clear selection';
  clearButton.hidden = true;
  const retryButton = document.createElement('button');
  retryButton.type = 'button';
  retryButton.textContent = 'Retry national geography';
  retryButton.hidden = true;
  geographyActions.appendChild(clearButton);
  geographyActions.appendChild(retryButton);
  const geographyActionHost = jurisdictionPicker?.parentNode || stage;
  if (geographyActionHost?.appendChild) geographyActionHost.appendChild(geographyActions);
  clearButton.addEventListener('click', () => {
    clearSelection();
    if (typeof jurisdictionPicker?.focus === 'function') jurisdictionPicker.focus();
  });
  retryButton.addEventListener('click', () => { void loadNationalGeography({ force: true }); });

  function syncJurisdictionPicker(code) {
    if (jurisdictionPicker) jurisdictionPicker.value = code || "";
  }

  function syncPublicLocation(state) {
    const code = window.ElectionPulsePublicCoverage?.normalizeState(state);
    const url = new URL(window.location.href);
    if (code) url.searchParams.set('geo_state', code);
    else url.searchParams.delete('geo_state');
    const next = `${url.pathname}${url.search}${url.hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== current) window.history.replaceState(window.history.state, '', next);
  }

  function selectPublicCoverage(state) {
    const control = window.ElectionPulsePublicGeoControl;
    if (control) void control.select(state);
  }


  function setText(node, value) {
    if (node) {
      node.textContent = value;
    }
  }

  function setGeographicContext(jurisdiction, selected) {
    if (!jurisdiction) {
      setText(contextTitle, "Select a jurisdiction");
      setText(
        contextCopy,
        "Hover, focus, or select a jurisdiction. Geographic selection does not imply ElectionPulse result availability."
      );
      setText(contextCode, "56 geographic jurisdictions");
      setText(contextStatus, "Election values intentionally absent at this layer");

      if (!selectedJurisdiction) {
        setText(lensTitle, "United States + Territories");
        setText(
          lensCopy,
          "Earth is the visual origin. Select a jurisdiction above to narrow geographic focus without inventing election evidence."
        );
      }
      return;
    }

    const kind =
      jurisdiction.kind === "territory"
        ? "Territory"
        : jurisdiction.kind === "district"
          ? "Federal district"
          : "State";

    const selectedElsewhere =
      Boolean(selectedJurisdiction) &&
      selectedJurisdiction.geoid !== jurisdiction.geoid;

    setText(contextTitle, jurisdiction.name);
    setText(
      contextCopy,
      selected
        ? `${kind} geographic lens locked from the governed Census 2025 national layer.`
        : `${kind} geographic preview from the governed Census 2025 national layer.`
    );
    setText(
      contextCode,
      `${jurisdiction.abbr} \u00B7 GEOID ${jurisdiction.geoid} \u00B7 ${kind}`
    );
    setText(
      contextStatus,
      selected
        ? "Lens locked \u00B7 no availability implied \u00B7 activate again or Esc to release"
        : selectedElsewhere
          ? `Preview only \u00B7 ${selectedJurisdiction.abbr} remains locked`
          : "Preview geography \u00B7 click or press Enter to lock lens"
    );

    if (selected) {
      setText(lensTitle, jurisdiction.name);
      setText(
        lensCopy,
        `${jurisdiction.abbr} is the current geographic lens. Election values remain governed separately and are not inferred by map selection.`
      );
    } else if (selectedElsewhere) {
      setText(lensTitle, `${selectedJurisdiction.name} · Locked`);
      setText(
        lensCopy,
        `Previewing ${jurisdiction.name}. ${selectedJurisdiction.abbr} remains the locked geographic lens; no election availability is inferred.`
      );
    } else {
      setText(lensTitle, `${jurisdiction.name} · Preview`);
      setText(
        lensCopy,
        `${jurisdiction.abbr} is a geographic preview only. Click or press Enter to lock this lens without implying election-result availability.`
      );
    }
  }

  function displayPoint(jurisdiction) {
    const point = Array.isArray(jurisdiction.label_point)
      ? jurisdiction.label_point
      : [0, 0];

    const x = Number(point[0]) || 0;
    const sourceY = Number(point[1]) || 0;
    const y = jurisdiction.display_group === "conus"
      ? CONUS_REFLECT_Y - sourceY
      : sourceY;

    return [x, y];
  }

  function isMicroJurisdiction(jurisdiction) {
    if (!Array.isArray(jurisdiction.bbox) || jurisdiction.bbox.length !== 4) {
      return false;
    }

    const width = Math.abs(Number(jurisdiction.bbox[2]) - Number(jurisdiction.bbox[0]));
    const height = Math.abs(Number(jurisdiction.bbox[3]) - Number(jurisdiction.bbox[1]));

    return (
      jurisdiction.display_group === "conus" &&
      (width < 18 || height < 18)
    );
  }

  function needsPointerAssist(jurisdiction) {
    return isMicroJurisdiction(jurisdiction) || jurisdiction.abbr === "HI";
  }

  function ensureSignalTrace() {
    if (!mapSignal) {
      return null;
    }

    if (signalTrace) {
      return signalTrace;
    }

    const namespace = "http://www.w3.org/2000/svg";
    signalTrace = document.createElementNS(namespace, "path");
    signalTrace.classList.add("observatory-map-signal-trace");
    mapSignal.insertBefore(signalTrace, mapSignal.firstChild);
    return signalTrace;
  }


  function setSignal(jurisdiction, active, mode = "preview") {
    if (!mapSignal) {
      return;
    }

    mapSignal.classList.remove("is-preview-signal", "is-selected-signal");

    if (!jurisdiction || !active) {
      mapSignal.classList.remove("is-active");
      delete mapSignal.dataset.mode;
      return;
    }

    const [x, y] = displayPoint(jurisdiction);
    const trace = ensureSignalTrace();
    const anchorX = 500;
    const anchorY = 610;
    const dx = anchorX - x;
    const dy = anchorY - y;
    const controlX = dx * 0.58;
    const controlY = dy * 0.36;

    if (trace) {
      trace.setAttribute(
        "d",
        `M0 0 Q ${controlX.toFixed(2)} ${controlY.toFixed(2)} ${dx.toFixed(2)} ${dy.toFixed(2)}`
      );
    }

    mapSignal.setAttribute("transform", `translate(${x} ${y})`);
    mapSignal.dataset.mode = mode;
    mapSignal.classList.add(
      "is-active",
      mode === "selected" ? "is-selected-signal" : "is-preview-signal"
    );
  }

  function setContextEngaged(engaged) {
    if (contextPanel) {
      contextPanel.classList.toggle("is-engaged", engaged);
    }
  }

  function clearSelection() {
    if (selectedPath) {
      selectedPath.classList.remove("is-selected");
    }

    if (selectedFeature) {
      selectedFeature.classList.remove("is-selected-feature");
    }

    selectedPath = null;
    selectedFeature = null;
    selectedJurisdiction = null;
    clearButton.hidden = true;
    syncJurisdictionPicker(null);
    updateProjectPrefill();
    syncPublicLocation(null);
    window.ElectionPulsePublicGeoControl?.clear();
    delete shell.dataset.selectedGeoid;
    mapStage.classList.remove("has-selection");
    setSignal(null, false);
    setContextEngaged(false);

    setText(lensTitle, "United States + Territories");
    setText(
      lensCopy,
      "Earth is the visual origin. Select a jurisdiction above to narrow geographic focus without inventing election evidence."
    );
    setGeographicContext(null, false);
  }

  function selectJurisdiction(feature, path, jurisdiction) {
    if (selectedPath === path) {
      clearSelection();
      feature.classList.remove("is-previewed-feature");
      path.classList.remove("is-previewed");
      feature.classList.add("is-release-pending");
      path.classList.add("is-release-pending");
      setSignal(null, false);
      setContextEngaged(false);
      setGeographicContext(null, false);
      return;
    }

    feature.classList.remove("is-release-pending");
    path.classList.remove("is-release-pending");

    if (selectedPath && selectedPath !== path) {
      selectedPath.classList.remove("is-selected");
    }

    if (selectedFeature && selectedFeature !== feature) {
      selectedFeature.classList.remove("is-selected-feature");
    }

    selectedPath = path;
    selectedFeature = feature;
    selectedJurisdiction = jurisdiction;
    clearButton.hidden = false;
    syncJurisdictionPicker(jurisdiction.abbr);
    updateProjectPrefill();
    syncPublicLocation(jurisdiction.abbr);
    selectPublicCoverage(jurisdiction.abbr);
    path.classList.add("is-selected");
    feature.classList.add("is-selected-feature");
    shell.dataset.selectedGeoid = jurisdiction.geoid;
    mapStage.classList.add("has-selection");
    setSignal(jurisdiction, true, "selected");
    setContextEngaged(true);
    setGeographicContext(jurisdiction, true);
  }

  function createJurisdictionFeature(jurisdiction) {
    const namespace = "http://www.w3.org/2000/svg";
    const feature = document.createElementNS(namespace, "g");
    const path = document.createElementNS(namespace, "path");
    const label = document.createElementNS(namespace, "text");
    const [labelX, labelY] = displayPoint(jurisdiction);
    const micro = isMicroJurisdiction(jurisdiction);
    const pointerAssist = needsPointerAssist(jurisdiction);

    feature.classList.add("observatory-map-feature");
    feature.dataset.geoid = jurisdiction.geoid;
    feature.dataset.group = jurisdiction.display_group || "unknown";

    if (micro) {
      feature.classList.add("is-micro-feature");
    }

    if (pointerAssist) {
      feature.classList.add("has-pointer-assist");
    }

    path.setAttribute("d", jurisdiction.path);
    path.setAttribute("tabindex", "0");
    path.setAttribute("role", "button");
    path.setAttribute("aria-label", `${jurisdiction.name}, geographic focus`);
    path.dataset.geoid = jurisdiction.geoid;
    path.dataset.abbr = jurisdiction.abbr;
    path.classList.add("observatory-map-jurisdiction");

    // Census/Albers projection is north-positive, while SVG y grows downward.
    // Only CONUS used the Albers path. The Alaska/Hawaii/territory inset builder
    // already inverted latitude before fitting, so a global SVG flip would break them.
    if (jurisdiction.display_group === "conus") {
      path.setAttribute("transform", `translate(0 ${CONUS_REFLECT_Y}) scale(1 -1)`);
    }

    label.setAttribute("x", String(labelX));
    label.setAttribute("y", String(labelY));
    label.setAttribute("aria-hidden", "true");
    label.classList.add("observatory-map-label");
    label.textContent = jurisdiction.abbr;

    if (pointerAssist) {
      const hitTarget = document.createElementNS(namespace, "circle");
      hitTarget.setAttribute("cx", String(labelX));
      hitTarget.setAttribute("cy", String(labelY));
      hitTarget.setAttribute(
        "r",
        jurisdiction.abbr === "HI"
          ? "14"
          : jurisdiction.abbr === "DC"
            ? "12"
            : "10"
      );
      hitTarget.setAttribute("aria-hidden", "true");
      hitTarget.classList.add("observatory-map-hit-target");
      feature.append(path, hitTarget, label);
    } else {
      feature.append(path, label);
    }

    function preview() {
      feature.classList.remove("is-release-pending");
      path.classList.remove("is-release-pending");
      feature.classList.add("is-previewed-feature");
      path.classList.add("is-previewed");
      setContextEngaged(true);
      setGeographicContext(jurisdiction, path === selectedPath);

      if (!selectedJurisdiction) {
        setSignal(jurisdiction, true, "preview");
      }
    }

    function releasePreview() {
      feature.classList.remove("is-previewed-feature", "is-release-pending");
      path.classList.remove("is-previewed", "is-release-pending");

      if (selectedJurisdiction) {
        setSignal(selectedJurisdiction, true, "selected");
        setContextEngaged(true);
        setGeographicContext(selectedJurisdiction, true);
      } else {
        setSignal(null, false);
        setContextEngaged(false);
        setGeographicContext(null, false);
      }
    }

    feature.addEventListener("mouseenter", preview);
    feature.addEventListener("mouseleave", releasePreview);
    feature.addEventListener("click", () => selectJurisdiction(feature, path, jurisdiction));

    path.addEventListener("focus", preview);
    path.addEventListener("blur", releasePreview);
    path.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectJurisdiction(feature, path, jurisdiction);
      } else if (event.key === "Escape") {
        event.preventDefault();
        clearSelection();
        path.blur();
      }
    });

    return feature;
  }

  function showGeographyUnavailable(message) {
    mapStage.classList.remove('map-ready');
    mapStage.classList.add("map-error");
    retryButton.hidden = false;
    retryButton.disabled = false;
    clearButton.hidden = true;
    mapLoading?.setAttribute('aria-busy', 'false');
    if (jurisdictionPicker) {
      const unavailable = document.createElement("option");
      unavailable.value = "";
      unavailable.textContent = "Geographic choices unavailable";
      jurisdictionPicker.replaceChildren(unavailable);
      jurisdictionPicker.disabled = true;
      jurisdictionPickerEntries = new Map();
    }
    setText(mapLoading, message);
  }

  async function loadNationalGeography({ force = false } = {}) {
    if (!map || !mapStage || !mapPaths) return;
    geographySequence += 1;
    const ticket = geographySequence;
    if (geographyActive) geographyActive.abort();
    const controller = new AbortController();
    geographyActive = controller;
    retryButton.hidden = true;
    retryButton.disabled = true;
    mapStage.classList.remove('map-error');
    mapLoading?.setAttribute('aria-busy', 'true');
    setText(mapLoading, 'Loading governed national geography…');
    const source = map.dataset.mapSource;
    if (!source) {
      geographyActive = null;
      showGeographyUnavailable("National geography source is not configured.");
      return;
    }

    let timer = null;
    try {
      const deadline = new Promise((_, reject) => {
        timer = window.setTimeout(() => {
          const error = new Error('Geography read timed out');
          error.name = 'TimeoutError';
          reject(error);
          controller.abort();
        }, 12000);
      });
      const read = (async () => {
        const response = await fetch(source, {
          method: 'GET', credentials: 'same-origin', redirect: 'error',
          cache: force ? 'reload' : 'force-cache', signal: controller.signal,
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`Geography request failed (${response.status})`);
        return await response.json();
      })();
      const data = await Promise.race([read, deadline]);
      if (ticket !== geographySequence) return;

      if (
        data.jurisdiction_count !== 56 ||
        !Array.isArray(data.jurisdictions) ||
        data.jurisdictions.length !== 56
      ) {
        throw new Error("Governed geography contract requires exactly 56 jurisdictions.");
      }

      if (
        !data.semantics ||
        data.semantics.contains_election_values !== false ||
        data.semantics.contains_data_availability !== false ||
        data.semantics.contains_support_status !== false ||
        data.semantics.selection_meaning !== "GEOGRAPHIC_FOCUS_ONLY"
      ) {
        throw new Error("Governed geography semantics are invalid.");
      }

      const fragment = document.createDocumentFragment();
      const pickerEntries = new Map();

      data.jurisdictions.forEach((jurisdiction) => {
        const feature = createJurisdictionFeature(jurisdiction);
        fragment.appendChild(feature);
        if (jurisdictionPicker) {
          const code = jurisdiction.abbr;
          if (typeof code !== "string" || !/^[A-Z]{2}$/.test(code) || pickerEntries.has(code)) {
            throw new Error("Geographic selector codes must be unique two-letter values.");
          }
          pickerEntries.set(code, {
            jurisdiction,
            feature,
            path: feature.querySelector("path[data-abbr]")
          });
        }
      });

      if (ticket !== geographySequence) return;
      // Remove stale selection references before replacing the map paths.
      if (selectedJurisdiction) clearSelection();
      mapPaths.replaceChildren(fragment);
      if (jurisdictionPicker) {
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Select a state or territory";
        const options = Array.from(pickerEntries.entries())
          .sort((a, b) => a[1].jurisdiction.name.localeCompare(b[1].jurisdiction.name))
          .map(([code, entry]) => {
            const option = document.createElement("option");
            option.value = code;
            option.textContent = `${entry.jurisdiction.name} (${code})`;
            return option;
          });
        jurisdictionPicker.replaceChildren(placeholder, ...options);
        jurisdictionPickerEntries = pickerEntries;
        jurisdictionPicker.disabled = false;
      }
      mapStage.classList.add("map-ready");
      retryButton.hidden = true;
      retryButton.disabled = false;
      mapLoading?.setAttribute('aria-busy', 'false');
      if (force && typeof jurisdictionPicker?.focus === 'function') jurisdictionPicker.focus();
      mapStage.dataset.orientation = "north-up";
      mapStage.dataset.instrumentMode = "geographic-focus";
      mapStage.dataset.microHitTargets = String(
        data.jurisdictions.filter(isMicroJurisdiction).length
      );
      mapStage.dataset.pointerAssistTargets = String(
        data.jurisdictions.filter(needsPointerAssist).length
      );
      setGeographicContext(null, false);
      // Restore display intent after route changes/back/reload; server data is revalidated.
      const urlState = window.ElectionPulsePublicCoverage?.normalizeState(
        new URLSearchParams(window.location.search).get('geo_state')
      );
      const initial = data.jurisdictions.find((item) => item.abbr === urlState);
      if (initial) {
        const path = Array.from(mapPaths.querySelectorAll('path[data-abbr]'))
          .find((item) => item.dataset.abbr === urlState);
        if (path) selectJurisdiction(path.parentNode, path, initial);
      }
    } catch (error) {
      if (ticket !== geographySequence) return;
      console.error("ElectionPulse Observatory geography load failed.", error);
      showGeographyUnavailable(error?.name === 'TimeoutError'
        ? 'Governed national geography timed out. No election values are inferred.'
        : 'Governed national geography could not be resolved. Election values remain unavailable.');
    } finally {
      if (timer !== null) window.clearTimeout(timer);
      if (ticket === geographySequence) geographyActive = null;
    }
  }

  function initMapInstrumentControls() {
    if (!map || !mapStage) {
      return;
    }

    map.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && selectedJurisdiction) {
        event.preventDefault();
        clearSelection();
      }
    });
  }


  function initPathwayFocus() {
    pathways.forEach((pathway) => {
      const key = pathway.dataset.observatoryPathway;

      function activate() {
        shell.dataset.activePathway = key;
      }

      function release() {
        delete shell.dataset.activePathway;
      }

      pathway.addEventListener("mouseenter", activate);
      pathway.addEventListener("focus", activate);
      pathway.addEventListener("mouseleave", release);
      pathway.addEventListener("blur", release);
    });
  }

  function initFocusWeight() {
    if (!stage) {
      return;
    }

    function focusData() {
      shell.dataset.observatoryFocus = "data";
    }

    function releaseData() {
      delete shell.dataset.observatoryFocus;
    }

    stage.addEventListener("mouseenter", focusData);
    stage.addEventListener("mouseleave", releaseData);
    stage.addEventListener("focusin", focusData);
    stage.addEventListener("focusout", releaseData);
  }

  function initPhraseCycle() {
    if (!phrase || reducedMotion.matches) {
      return;
    }

    window.setInterval(() => {
      phraseIndex = (phraseIndex + 1) % phrases.length;
      phrase.textContent = phrases[phraseIndex];
    }, 7200);
  }

  function init() {
    if (jurisdictionPicker) {
      jurisdictionPicker.addEventListener("change", () => {
        const entry = jurisdictionPickerEntries.get(jurisdictionPicker.value);
        if (!entry) {
          clearSelection();
          return;
        }
        if (selectedPath !== entry.path) {
          selectJurisdiction(entry.feature, entry.path, entry.jurisdiction);
        }
      });
    }
    initPathwayFocus();
    initFocusWeight();
    initMapInstrumentControls();
    initPhraseCycle();
    loadNationalGeography();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
