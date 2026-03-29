const DATA_DIR = "data/";

const DEFAULT_STAGE_NAMES = {
    "1": "S1: Dense Mangrove",
    "2": "S2: Degradation",
    "3": "S3: Clearing",
    "4": "S4: Water Filling",
    "5": "S5: Operational Pond",
};

const DEFAULT_STAGE_COLORS = {
    "1": "#1a5e2a",
    "2": "#c9b42c",
    "3": "#d97b28",
    "4": "#28b5d9",
    "5": "#1a3fa0",
};

let timelineData = null;
let alertData = null;
let statsData = null;
let accuracyData = null;
let polygonData = null;
let leafletMap = null;
let availableYears = [];

document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".nav-btn").forEach((node) => node.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((node) => node.classList.remove("active"));

        btn.classList.add("active");
        const tabId = `tab-${btn.dataset.tab}`;
        const tabNode = document.getElementById(tabId);
        if (tabNode) {
            tabNode.classList.add("active");
        }

        if (btn.dataset.tab === "map" && !leafletMap) {
            initMap();
        }
    });
});

async function loadJSON(filename) {
    try {
        const response = await fetch(DATA_DIR + filename);
        if (!response.ok) {
            return null;
        }
        return await response.json();
    } catch (error) {
        console.log(`[INFO] ${filename} not yet generated. Run the pipeline first.`);
        return null;
    }
}

function normalizePolygonData(data) {
    if (!data) {
        return null;
    }
    if (data.type === "FeatureCollection" && Array.isArray(data.features)) {
        return data;
    }
    if (Array.isArray(data.features)) {
        return { type: "FeatureCollection", features: data.features };
    }
    if (!Array.isArray(data.ponds)) {
        return null;
    }

    const features = data.ponds
        .map((pond) => {
            const geometry = pond.geometry || (
                pond.lon != null && pond.lat != null
                    ? { type: "Point", coordinates: [pond.lon, pond.lat] }
                    : null
            );
            if (!geometry) {
                return null;
            }
            const stage = pond.confirmed_stage || pond.raw_stage || 0;
            return {
                type: "Feature",
                geometry,
                properties: {
                    pond_id: pond.pond_id || null,
                    date: pond.date || null,
                    stage,
                    stage_code: stage > 0 ? `S${stage}` : "S0",
                    stage_name: stage > 0 ? getStageNames()[String(stage)] : "Unclassified",
                    confidence: pond.confidence ?? null,
                    area_m2: pond.area_m2 ?? null,
                    rectangularity: pond.rectangularity ?? null,
                    lat: pond.lat ?? null,
                    lon: pond.lon ?? null,
                },
            };
        })
        .filter(Boolean);

    return { type: "FeatureCollection", features };
}

async function loadPolygonData() {
    const geojson = normalizePolygonData(await loadJSON("polygons.geojson"));
    if (geojson) {
        return geojson;
    }
    return normalizePolygonData(await loadJSON("polygons.json"));
}

function getStatsRecords() {
    if (statsData?.yearly_stats?.length) {
        return statsData.yearly_stats;
    }
    if (statsData?.image_stats?.length) {
        return statsData.image_stats;
    }
    return [];
}

function getStageNames() {
    return timelineData?.stage_names || statsData?.stage_names || DEFAULT_STAGE_NAMES;
}

function getStageColors() {
    return timelineData?.stage_colors || statsData?.stage_colors || DEFAULT_STAGE_COLORS;
}

function deriveAvailableYears() {
    const years = new Set();

    (timelineData?.available_years || []).forEach((year) => years.add(Number(year)));
    getStatsRecords().forEach((record) => {
        if (record?.year != null) {
            years.add(Number(record.year));
        }
    });
    (timelineData?.timeline || []).forEach((entry) => {
        if (entry?.year != null) {
            years.add(Number(entry.year));
        }
    });

    return Array.from(years)
        .filter((year) => Number.isFinite(year))
        .sort((left, right) => left - right);
}

function getRecordByYear(year) {
    const records = getStatsRecords();
    return records.find((record) => Number(record.year) === Number(year)) || null;
}

function getFallbackAsset(kind, year) {
    const paths = {
        rgb: `../outputs/images/rgb_${year}.png`,
        detection: `../outputs/images/detection_${year}.png`,
        ground_truth: `../outputs/images/ground_truth_${year}.png`,
        stage: `../outputs/images/stage_${year}.png`,
        ndvi: `../outputs/features/ndvi_${year}.png`,
        mndwi: `../outputs/features/mndwi_${year}.png`,
        ndwi: `../outputs/features/ndwi_${year}.png`,
        savi: `../outputs/features/savi_${year}.png`,
        ndbi: `../outputs/features/ndbi_${year}.png`,
        awei: `../outputs/features/awei_${year}.png`,
        sar: `../outputs/features/sar_${year}.png`,
        rvi: `../outputs/features/rvi_${year}.png`,
    };
    return paths[kind] || "";
}

function getAssetForYear(year, kind) {
    const record = getRecordByYear(year);
    return record?.assets?.[kind] || getFallbackAsset(kind, year);
}

async function loadAllData() {
    [
        timelineData,
        alertData,
        statsData,
        accuracyData,
        polygonData,
    ] = await Promise.all([
        loadJSON("timeline.json"),
        loadJSON("alerts.json"),
        loadJSON("stats.json"),
        loadJSON("accuracy.json"),
        loadPolygonData(),
    ]);

    availableYears = deriveAvailableYears();

    renderAlerts();
    renderTimeline();
    renderAnalytics();
    renderAccuracy();
    setupSliders();
    updateStatus();
}

function updateStatus() {
    const node = document.getElementById("system-status");
    if (!node) {
        return;
    }

    const recordCount = getStatsRecords().length;
    if (recordCount > 0) {
        node.textContent = `● ${recordCount} epochs processed`;
        node.style.color = "#10b981";
        return;
    }

    node.textContent = "○ Awaiting pipeline run";
    node.style.color = "#5a6580";
}

function renderAlerts() {
    const container = document.getElementById("alert-list");
    if (!container) {
        return;
    }

    if (!alertData?.alerts?.length) {
        container.innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">Alerts</div>
                <p>No alerts yet. Run the pipeline to detect stage transitions.</p>
                <p style="margin-top:0.5rem; font-size:0.78rem">python main.py</p>
            </div>`;
        return;
    }

    container.innerHTML = alertData.alerts.map((alert) => {
        const confidence = Number(alert.confidence || 0);
        const confidenceClass = confidence >= 0.85 ? "high" : "medium";
        return `
            <div class="alert-item">
                <div class="alert-icon transition">!</div>
                <div class="alert-body">
                    <div class="alert-title">${alert.old_stage_name} -> ${alert.new_stage_name}</div>
                    <div class="alert-detail">Transition confirmed for ${alert.date}</div>
                </div>
                <span class="alert-confidence ${confidenceClass}">${(confidence * 100).toFixed(1)}%</span>
                <span class="alert-date">${alert.date}</span>
            </div>`;
    }).join("");
}

function renderTimeline() {
    const container = document.getElementById("timeline-container");
    if (!container) {
        return;
    }

    if (!timelineData?.timeline?.length) {
        container.innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">Timeline</div>
                <p>No timeline data. Run the pipeline to generate stage history.</p>
            </div>`;
        return;
    }

    const stageNames = getStageNames();
    const stageColors = getStageColors();

    container.innerHTML = timelineData.timeline.map((entry) => {
        const stageId = entry.confirmed_stage || entry.stage || 0;
        const stageKey = String(stageId);
        const stageClass = `s${stageId}`;
        const stageName = entry.confirmed_stage_name || entry.stage_name || stageNames[stageKey] || "Unknown";
        const dotColor = stageColors[stageKey] || "#666666";
        const uncertaintyBadge = entry.uncertain
            ? `<span class="timeline-conf" title="${entry.uncertainty_reason || "Low certainty"}">Uncertain</span>`
            : "";
        return `
            <div class="timeline-entry ${stageClass}">
                <div class="timeline-dot" style="background:${dotColor}"></div>
                <span class="timeline-date">${entry.date}</span>
                <span class="timeline-stage stage-${stageClass}">${stageName}</span>
                <span class="timeline-conf">Confidence: ${(Number(entry.confidence || 0) * 100).toFixed(1)}%</span>
                ${uncertaintyBadge}
            </div>`;
    }).join("");
}

function configureYearSlider(sliderId, labelId, onYearChange) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    if (!slider || !label) {
        return;
    }

    if (!availableYears.length) {
        slider.disabled = true;
        label.textContent = "N/A";
        return;
    }

    slider.disabled = false;
    slider.min = "0";
    slider.max = String(availableYears.length - 1);
    slider.step = "1";
    slider.value = String(availableYears.length - 1);

    const sync = () => {
        const index = Math.max(0, Math.min(availableYears.length - 1, Number(slider.value) || 0));
        const year = availableYears[index];
        label.textContent = String(year);
        onYearChange(year);
    };

    slider.oninput = sync;
    sync();
}

function setupSliders() {
    configureYearSlider("year-slider", "year-label", updateSatelliteImages);
    configureYearSlider("feature-year-slider", "feature-year-label", updateFeatureImages);
}

function updateSatelliteImages(year) {
    setImage("img-raw", getAssetForYear(year, "rgb"));
    setImage("img-detection", getAssetForYear(year, "detection"));
}

function updateFeatureImages(year) {
    setImage("img-ndvi", getAssetForYear(year, "ndvi"));
    setImage("img-mndwi", getAssetForYear(year, "mndwi"));
    setImage("img-sar", getAssetForYear(year, "sar"));
    setImage("img-stage-feat", getAssetForYear(year, "stage"));
}

function setImage(containerId, src) {
    const container = document.getElementById(containerId);
    const img = container?.querySelector("img");
    if (!container || !img) {
        return;
    }

    if (!src) {
        img.removeAttribute("src");
        img.style.display = "none";
        return;
    }

    img.style.display = "none";
    img.src = src;
    img.onerror = () => {
        img.style.display = "none";
    };
    img.onload = () => {
        img.style.display = "block";
    };
}

function initMap() {
    const aoi = statsData?.aoi || timelineData?.aoi || {};
    const center = [
        Number(aoi.center_lat ?? 16.65),
        Number(aoi.center_lon ?? 82.225),
    ];
    const zoom = Number(aoi.zoom ?? 12);

    leafletMap = L.map("map").setView(center, zoom);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 18,
    }).addTo(leafletMap);

    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        attribution: "© Esri",
        maxZoom: 18,
    }).addTo(leafletMap);

    const bbox = aoi.bbox || [82.10, 16.55, 82.35, 16.75];
    L.rectangle([
        [bbox[1], bbox[0]],
        [bbox[3], bbox[2]],
    ], {
        color: "#06b6d4",
        weight: 2,
        fillOpacity: 0.05,
        dashArray: "8 4",
    }).addTo(leafletMap).bindPopup(`Study AOI: ${aoi.name || "Godavari Delta"}`);

    if (polygonData?.features?.length) {
        const colors = getStageColors();
        L.geoJSON(polygonData, {
            style: (feature) => {
                const stageValue = feature?.properties?.stage ?? 0;
                const color = colors[String(stageValue)] || colors[String(String(stageValue).replace(/^S/, ""))] || "#28b5d9";
                return {
                    color,
                    weight: 2,
                    fillOpacity: 0.25,
                };
            },
            pointToLayer: (feature, latlng) => {
                const stageValue = feature?.properties?.stage ?? 0;
                const color = colors[String(stageValue)] || "#28b5d9";
                return L.circleMarker(latlng, {
                    radius: 6,
                    color,
                    fillColor: color,
                    fillOpacity: 0.8,
                    weight: 2,
                });
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties || {};
                const stageName = props.stage_name || props.stage_code || "Unknown";
                const confidence = props.confidence != null ? `${(Number(props.confidence) * 100).toFixed(1)}%` : "N/A";
                layer.bindPopup(`
                    <strong>${props.pond_id || "Pond"}</strong><br>
                    Stage: ${stageName}<br>
                    Date: ${props.date || "N/A"}<br>
                    Confidence: ${confidence}<br>
                    Uncertain: ${props.uncertain ? "Yes" : "No"}<br>
                    Reason: ${props.uncertainty_reason || "N/A"}<br>
                    Rectangularity: ${props.rectangularity != null ? Number(props.rectangularity).toFixed(2) : "N/A"}
                `);
            },
        }).addTo(leafletMap);
    }

    setTimeout(() => leafletMap.invalidateSize(), 200);
}

function renderAnalytics() {
    const stats = getStatsRecords();
    if (!stats.length || typeof Chart === "undefined") {
        return;
    }

    const labels = stats.map((record) => record.date);
    const stageColors = getStageColors();

    Chart.defaults.color = "#8b96b0";
    Chart.defaults.borderColor = "rgba(42, 53, 85, 0.5)";
    Chart.defaults.font.family = "'Inter', sans-serif";

    new Chart(document.getElementById("chart-ndvi"), {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Mean NDVI",
                data: stats.map((record) => record.ndvi_mean || 0),
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: "#10b981",
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { title: { display: true, text: "NDVI" } },
            },
        },
    });

    new Chart(document.getElementById("chart-mndwi"), {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Mean MNDWI",
                data: stats.map((record) => record.mndwi_mean || 0),
                borderColor: "#06b6d4",
                backgroundColor: "rgba(6, 182, 212, 0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: "#06b6d4",
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { title: { display: true, text: "MNDWI" } },
            },
        },
    });

    new Chart(document.getElementById("chart-stages"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Stage",
                data: stats.map((record) => record.confirmed_stage || record.stage || 0),
                backgroundColor: stats.map((record) => {
                    const stageValue = record.confirmed_stage || record.stage || 0;
                    return stageColors[String(stageValue)] || "#666666";
                }),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 0,
                    max: 6,
                    title: { display: true, text: "Stage (1-5)" },
                    ticks: {
                        stepSize: 1,
                        callback: (value) => ["", "S1", "S2", "S3", "S4", "S5"][value] || "",
                    },
                },
            },
        },
    });

    new Chart(document.getElementById("chart-confidence"), {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Confidence",
                data: stats.map((record) => Number(record.confidence || 0) * 100),
                borderColor: "#8b5cf6",
                backgroundColor: "rgba(139, 92, 246, 0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: "#8b5cf6",
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: "Confidence %" },
                },
            },
        },
    });
}

function renderAccuracy() {
    const container = document.getElementById("accuracy-grid");
    if (!container) {
        return;
    }

    if (!accuracyData) {
        container.innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">Accuracy</div>
                <p>No accuracy data yet. Run the pipeline to generate validation results.</p>
            </div>`;
        return;
    }

    let html = "";

    if (accuracyData.gmw) {
        const gmw = accuracyData.gmw;
        html += `
            <div class="accuracy-card">
                <h3>Global Mangrove Watch Comparison</h3>
                ${gmw.gmw_available ? `
                    <div class="metric-row"><span class="metric-label">Overall Accuracy</span><span class="metric-value">${(gmw.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">Precision</span><span class="metric-value">${(gmw.precision * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">Recall</span><span class="metric-value">${(gmw.recall * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">F1 Score</span><span class="metric-value">${(gmw.f1_score * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">True Positives</span><span class="metric-value">${gmw.true_positive}</span></div>
                    <div class="metric-row"><span class="metric-label">False Positives</span><span class="metric-value">${gmw.false_positive}</span></div>
                ` : `<p style="color:#5a6580">GMW baseline not available</p>`}
            </div>`;
    }

    if (accuracyData.ground_truth) {
        const gt = accuracyData.ground_truth;
        html += `
            <div class="accuracy-card">
                <h3>Detection vs Ground Truth</h3>
                ${gt.available ? `
                    <div class="metric-row"><span class="metric-label">Precision</span><span class="metric-value">${(Number(gt.precision || 0) * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">Recall</span><span class="metric-value">${(Number(gt.recall || 0) * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">F1 Score</span><span class="metric-value">${(Number(gt.f1_score || 0) * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">GT Matched</span><span class="metric-value">${gt.gt_matched || 0}/${gt.gt_total || 0}</span></div>
                    <div class="metric-row"><span class="metric-label">Detections Matched</span><span class="metric-value">${gt.det_matched || 0}/${gt.det_total || 0}</span></div>
                ` : `<p style="color:#5a6580">Ground-truth comparison not available yet</p>`}
            </div>`;
    }

    if (accuracyData.jrc) {
        const jrc = accuracyData.jrc;
        html += `
            <div class="accuracy-card">
                <h3>JRC Water Comparison</h3>
                ${jrc.jrc_available ? `
                    <div class="metric-row"><span class="metric-label">Overall Accuracy</span><span class="metric-value">${(jrc.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="metric-row"><span class="metric-label">True Positives</span><span class="metric-value">${jrc.true_positive}</span></div>
                    <div class="metric-row"><span class="metric-label">False Positives</span><span class="metric-value">${jrc.false_positive}</span></div>
                    <div class="metric-row"><span class="metric-label">False Negatives</span><span class="metric-value">${jrc.false_negative}</span></div>
                ` : `<p style="color:#5a6580">JRC data not available</p>`}
            </div>`;
    }

    if (accuracyData.confusion_matrix) {
        html += `
            <div class="accuracy-card">
                <h3>Confusion Matrix</h3>
                ${accuracyData.confusion_matrix.available
                    ? `<div class="metric-row"><span class="metric-label">Overall Accuracy</span><span class="metric-value">${(accuracyData.confusion_matrix.overall_accuracy * 100).toFixed(1)}%</span></div>
                       <div class="metric-row"><span class="metric-label">Kappa</span><span class="metric-value">${accuracyData.confusion_matrix.kappa.toFixed(4)}</span></div>`
                    : `<p style="color:#5a6580">${accuracyData.confusion_matrix.note || "Not yet computed"}</p>`}
            </div>`;
    }

    if (Array.isArray(accuracyData.comparison_panels) && accuracyData.comparison_panels.length) {
        const comparisonItems = accuracyData.comparison_panels
            .filter((panel) => panel?.path)
            .map((panel) => `
                <div class="image-card">
                    <h3>${panel.label}</h3>
                    <div class="image-placeholder">
                        <img src="${panel.path}" alt="${panel.label}" />
                    </div>
                </div>`)
            .join("");
        if (comparisonItems) {
            html += `
                <div class="accuracy-card" style="grid-column: 1 / -1;">
                    <h3>Visual Comparison Panels</h3>
                    <div class="image-grid">${comparisonItems}</div>
                </div>`;
        }
    }

    if (accuracyData.feature_audit) {
        const audit = accuracyData.feature_audit;
        const unused = Array.isArray(audit.unused_features) ? audit.unused_features : [];
        html += `
            <div class="accuracy-card">
                <h3>Feature Audit</h3>
                <div class="metric-row"><span class="metric-label">Computed Features</span><span class="metric-value">${audit.total_features || 0}</span></div>
                <div class="metric-row"><span class="metric-label">Unused Features</span><span class="metric-value">${unused.length}</span></div>
                <p style="margin-top:0.8rem; color:#5a6580">${unused.length ? unused.join(", ") : "All audited features are connected into the active pipeline."}</p>
            </div>`;
    }

    container.innerHTML = html || `<div class="no-data"><p>No accuracy data available</p></div>`;
}

document.addEventListener("DOMContentLoaded", loadAllData);
