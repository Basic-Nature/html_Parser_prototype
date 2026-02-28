let allMetrics = [];
let charts = {};
let integrityData = null;
let integrityCharts = {};
let dismissedAlerts = new Set();
let thresholds = {
    confDropThreshold: 0.08,
    unknownSpikeThreshold: 0.10,
    reviewSpikeThreshold: 5.0,
    baselineWindow: 30,
    recentWindow: 5
};

// Load metrics from API
async function loadMetrics() {
    const params = new URLSearchParams();

    const handlerEl = document.getElementById('handlerFilter');
    const stateEl = document.getElementById('stateFilter');
    const minConfEl = document.getElementById('minConfidence');
    const limitEl = document.getElementById('limitFilter');
    const handler = (handlerEl instanceof HTMLInputElement || handlerEl instanceof HTMLSelectElement) ? handlerEl.value : '';
    const state = (stateEl instanceof HTMLInputElement || stateEl instanceof HTMLSelectElement) ? stateEl.value : '';
    const minConf = (minConfEl instanceof HTMLInputElement || minConfEl instanceof HTMLSelectElement) ? minConfEl.value : '';
    const limit = (limitEl instanceof HTMLInputElement || limitEl instanceof HTMLSelectElement) ? limitEl.value : '';

    if (handler) params.append('handler', handler);
    if (state) params.append('state', state);
    if (minConf) params.append('min_confidence', minConf);
    if (limit) params.append('limit', limit);

    const response = await fetch(`/api/quality_metrics?${params}`);
    const data = await response.json();
    allMetrics = data.metrics;

    updateDashboard();
}

// Update all dashboard components
function updateDashboard() {
    updateStats();
    updateCharts();
    updateTable();
    updateStateFilter();
}

// Update summary stats
function updateStats() {
    document.getElementById('totalCount').textContent = String(allMetrics.length);

    const confidences = allMetrics
        .map(m => m.quality_metrics?.extraction_confidence)
        .filter(c => c != null);
    const avgConf = confidences.length > 0
        ? (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(3)
        : 'N/A';
    document.getElementById('avgConfidence').textContent = String(avgConf);

    const rows = allMetrics
        .map(m => m.row_count)
        .filter(r => r != null);
    const avgRows = rows.length > 0
        ? Math.round(rows.reduce((a, b) => a + b, 0) / rows.length)
        : 'N/A';
    document.getElementById('avgRows').textContent = String(avgRows);

    const cols = allMetrics
        .map(m => m.column_count)
        .filter(c => c != null);
    const avgCols = cols.length > 0
        ? Math.round(cols.reduce((a, b) => a + b, 0) / cols.length)
        : 'N/A';
    document.getElementById('avgCols').textContent = String(avgCols);
}

// Update charts
function updateCharts() {
    // Confidence over time
    const labels = allMetrics.map(m => m.timestamp.slice(0, 8)); // YYYYMMDD
    const confidences = allMetrics.map(m => m.quality_metrics?.extraction_confidence || 0);

    if (charts.confidence) charts.confidence.destroy();
    charts.confidence = new (/** @type {any} */ (window).Chart)(document.getElementById('confidenceChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Confidence',
                data: confidences,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0, max: 1 }
            }
        }
    });

    // Quality by handler
    const handlerGroups = {};
    allMetrics.forEach(m => {
        const handler = m.handler || 'unknown';
        if (!handlerGroups[handler]) handlerGroups[handler] = [];
        const conf = m.quality_metrics?.extraction_confidence;
        if (conf != null) handlerGroups[handler].push(conf);
    });

    const handlerLabels = Object.keys(handlerGroups);
    const handlerAvgs = handlerLabels.map(h => {
        const confs = handlerGroups[h];
        return confs.length > 0 ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    });

    if (charts.handler) charts.handler.destroy();
    charts.handler = new (/** @type {any} */ (window).Chart)(document.getElementById('handlerChart'), {
        type: 'bar',
        data: {
            labels: handlerLabels,
            datasets: [{
                label: 'Avg Confidence',
                data: handlerAvgs,
                backgroundColor: '#10b981'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0, max: 1 }
            }
        }
    });

    // Confidence distribution
    const bins = { high: 0, medium: 0, low: 0, unknown: 0 };
    allMetrics.forEach(m => {
        const conf = m.quality_metrics?.extraction_confidence;
        if (conf == null) bins.unknown++;
        else if (conf >= 0.8) bins.high++;
        else if (conf >= 0.5) bins.medium++;
        else bins.low++;
    });

    if (charts.distribution) charts.distribution.destroy();
    charts.distribution = new (/** @type {any} */ (window).Chart)(document.getElementById('distributionChart'), {
        type: 'doughnut',
        data: {
            labels: ['High (≥0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)', 'Unknown'],
            datasets: [{
                data: [bins.high, bins.medium, bins.low, bins.unknown],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#6b7280']
            }]
        },
        options: { responsive: true }
    });

    // Empty row ratio
    const emptyRatios = allMetrics.map(m =>
        m.quality_metrics?.empty_row_ratio != null
            ? m.quality_metrics.empty_row_ratio * 100
            : null
    ).filter(r => r != null);

    if (charts.emptyRow) charts.emptyRow.destroy();
    charts.emptyRow = new (/** @type {any} */ (window).Chart)(document.getElementById('emptyRowChart'), {
        type: 'line',
        data: {
            labels: labels.slice(0, emptyRatios.length),
            datasets: [{
                label: 'Empty Row %',
                data: emptyRatios,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0, max: 100 }
            }
        }
    });
}

// Update data table
function updateTable() {
    const tbody = document.getElementById('dataTableBody');
    // Clear children safely
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

    if (allMetrics.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 8;
        td.classList.add('table-empty-cell');
        td.textContent = 'No data';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }

    allMetrics.slice(0, 20).forEach(m => {
        const tr = document.createElement('tr');

        const makeTd = (text, opts = {}) => {
            const td = document.createElement('td');
            if (opts.maxWidth) {
                td.classList.add('td-truncate');
                td.classList.add('td-truncate-200');
            }
            td.textContent = text;
            return td;
        };

        tr.appendChild(makeTd(m.timestamp || ''));
        tr.appendChild(makeTd(m.handler || 'N/A'));
        tr.appendChild(makeTd(m.state || 'N/A'));
        tr.appendChild(makeTd(m.contest || 'N/A', { maxWidth: '200px' }));
        tr.appendChild(makeTd(m.row_count != null ? String(m.row_count) : 'N/A'));
        tr.appendChild(makeTd(m.column_count != null ? String(m.column_count) : 'N/A'));

        // Confidence badge
        const conf = m.quality_metrics?.extraction_confidence;
        const confTd = document.createElement('td');
        const confSpan = document.createElement('span');
        confSpan.classList.add('confidence-badge');
        if (conf == null) {
            confSpan.classList.add('confidence-unknown');
            confSpan.textContent = 'N/A';
        } else {
            const confPct = (conf * 100).toFixed(1) + '%';
            if (conf >= 0.8) confSpan.classList.add('confidence-high');
            else if (conf >= 0.5) confSpan.classList.add('confidence-medium');
            else confSpan.classList.add('confidence-low');
            confSpan.textContent = confPct;
        }
        confTd.appendChild(confSpan);
        tr.appendChild(confTd);

        const emptyRatio = m.quality_metrics?.empty_row_ratio;
        const emptyPct = emptyRatio != null ? (emptyRatio * 100).toFixed(1) + '%' : 'N/A';
        tr.appendChild(makeTd(emptyPct));

        tbody.appendChild(tr);
    });
}

// Update state filter dropdown
function updateStateFilter() {
    const stateFilter = document.getElementById('stateFilter');
    const currentValue = (stateFilter instanceof HTMLSelectElement || stateFilter instanceof HTMLInputElement) ? stateFilter.value : '';

    const states = new Set(allMetrics.map(m => m.state).filter(s => s));

    if (stateFilter instanceof HTMLSelectElement) {
        // Clear existing options
        while (stateFilter.firstChild) stateFilter.removeChild(stateFilter.firstChild);
        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.textContent = 'All States';
        stateFilter.appendChild(defaultOpt);
        Array.from(states).sort().forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            if (state === currentValue) option.selected = true;
            stateFilter.appendChild(option);
        });
    }
}

// Export dataset
document.getElementById('exportBtn').addEventListener('click', () => {
    const csvRows = [];
    const headers = ['timestamp', 'handler', 'state', 'county', 'contest', 'rows', 'cols', 'confidence', 'empty_ratio', 'null_ratio', 'avg_density'];
    csvRows.push(headers.join(','));

    allMetrics.forEach(m => {
        const q = m.quality_metrics || {};
        const row = [
            m.timestamp,
            m.handler || '',
            m.state || '',
            m.county || '',
            `"${(m.contest || '').replace(/"/g, '""')}"`,
            m.row_count || '',
            m.column_count || '',
            q.extraction_confidence || '',
            q.empty_row_ratio || '',
            q.null_cell_ratio || '',
            q.avg_row_density || ''
        ];
        csvRows.push(row.join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quality_metrics_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
});

// Event listeners
document.getElementById('handlerFilter').addEventListener('change', loadMetrics);
document.getElementById('stateFilter').addEventListener('change', loadMetrics);
document.getElementById('minConfidence').addEventListener('change', loadMetrics);
document.getElementById('limitFilter').addEventListener('change', loadMetrics);

// Initial load
loadMetrics();
loadIntegrityData();

// ============================================
// Integrity Monitoring Functions
// ============================================

// Load integrity trend data
async function loadIntegrityData() {
    try {
        // Load trend file
        const trendsResponse = await fetch('/api/integrity_trends');
        if (!trendsResponse.ok) {
            throw new Error(`Integrity trends request failed: ${trendsResponse.status}`);
        }
        const trendsData = await trendsResponse.json();
        
        // Compute current integrity signal
        const signalResponse = await fetch('/api/integrity_signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(thresholds)
        });
        if (!signalResponse.ok) {
            throw new Error(`Integrity signal request failed: ${signalResponse.status}`);
        }
        const signalData = await signalResponse.json();
        
        integrityData = {
            trends: trendsData.trends || [],
            signal: signalData.signal || {}
        };
        
        updateIntegrityDashboard();
    } catch (error) {
        console.error('Failed to load integrity data:', error);
        document.getElementById('integrityStatus').textContent = 'Error';
    }
}

// Update integrity dashboard components
function updateIntegrityDashboard() {
    updateIntegrityStats();
    updateAlerts();
    updateSparklines();
    updateComparisonSelects();
}

// Update integrity stats
function updateIntegrityStats() {
    const signal = integrityData.signal;
    const trends = integrityData.trends;
    
    // Status
    const statusEl = document.getElementById('integrityStatus');
    statusEl.textContent = signal.status || '-';
    statusEl.className = 'stat-value';
    if (signal.status === 'ok') statusEl.style.color = '#10b981';
    else if (signal.status === 'alert') statusEl.style.color = '#f59e0b';
    else if (signal.status === 'error') statusEl.style.color = '#ef4444';
    else statusEl.style.color = '#60a5fa';
    
    // Trend count
    document.getElementById('trendCount').textContent = signal.entry_count || trends.length || '-';
    
    // Alert count
    const alerts = (signal.alerts || []).filter(a => !dismissedAlerts.has(a.type));
    document.getElementById('alertCount').textContent = alerts.length;
    
    // Last analysis
    const lastEntry = trends.length > 0 ? trends[trends.length - 1] : null;
    const lastTimestamp = lastEntry?.timestamp || lastEntry?.generated_at;
    if (lastTimestamp) {
        const date = new Date(lastTimestamp);
        document.getElementById('lastAnalysis').textContent = date.toLocaleString();
    } else {
        document.getElementById('lastAnalysis').textContent = '-';
    }
}

// Update active alerts
function updateAlerts() {
    const alerts = (integrityData.signal.alerts || []).filter(a => !dismissedAlerts.has(a.type));
    const container = document.getElementById('alertsContainer');
    const listEl = document.getElementById('alertsList');
    
    if (alerts.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    listEl.innerHTML = '';
    
    alerts.forEach(alert => {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert-item';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'alert-content';
        
        const typeDiv = document.createElement('div');
        typeDiv.className = 'alert-type';
        const icons = {
            confidence_drop: '📉',
            unknown_spike: '❓',
            review_spike: '📋'
        };
        typeDiv.textContent = `${icons[alert.type] || '⚠️'} ${alert.type.replace(/_/g, ' ')}`;
        
        const msgDiv = document.createElement('div');
        msgDiv.className = 'alert-message';
        msgDiv.textContent = alert.message;
        
        contentDiv.appendChild(typeDiv);
        contentDiv.appendChild(msgDiv);
        
        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'alert-dismiss';
        dismissBtn.textContent = 'Dismiss';
        dismissBtn.onclick = () => dismissAlert(alert.type);
        
        alertDiv.appendChild(contentDiv);
        alertDiv.appendChild(dismissBtn);
        listEl.appendChild(alertDiv);
    });
}

// Dismiss alert
function dismissAlert(alertType) {
    dismissedAlerts.add(alertType);
    updateAlerts();
    updateIntegrityStats();
}

// Update sparkline charts
function updateSparklines() {
    const trends = integrityData.trends || [];
    if (trends.length === 0) return;
    
    const recent = integrityData.signal.recent || {};
    const deltas = integrityData.signal.deltas || {};
    
    // Confidence sparkline
    const confData = trends.map(t => t.confidence?.avg || 0);
    updateSparkline('confidenceSparkline', confData, '#a78bfa');
    document.getElementById('confCurrent').textContent = (recent.confidence_avg || 0).toFixed(3);
    updateDeltaDisplay('confDelta', deltas.confidence_avg_delta);
    
    // Unknown ratio sparkline
    const unknownData = trends.map(t => t.unknown_ratio || 0);
    updateSparkline('unknownSparkline', unknownData, '#f59e0b');
    document.getElementById('unknownCurrent').textContent = (recent.unknown_ratio || 0).toFixed(3);
    updateDeltaDisplay('unknownDelta', deltas.unknown_ratio_delta);
    
    // Segments review sparkline
    const reviewData = trends.map(t => t.review_signals?.segments_needing_review || 0);
    updateSparkline('reviewSparkline', reviewData, '#ef4444');
    document.getElementById('reviewCurrent').textContent = (recent.segments_review || 0).toFixed(1);
    updateDeltaDisplay('reviewDelta', deltas.segments_review_delta);
    
    // Pattern KB matches sparkline
    const kbData = trends.map(t => t.review_signals?.pattern_kb_matches || 0);
    updateSparkline('kbSparkline', kbData, '#10b981');
    document.getElementById('kbCurrent').textContent = (recent.pattern_kb_matches || 0).toFixed(1);
    updateDeltaDisplay('kbDelta', deltas.pattern_kb_matches_delta);
}

// Update single sparkline chart
function updateSparkline(canvasId, data, color) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    if (integrityCharts[canvasId]) {
        integrityCharts[canvasId].destroy();
    }
    
    integrityCharts[canvasId] = new (/** @type {any} */ (window).Chart)(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                data: data,
                borderColor: color,
                backgroundColor: `${color}20`,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

// Update delta display
function updateDeltaDisplay(elementId, delta) {
    const el = document.getElementById(elementId);
    if (!el || delta == null) {
        if (el) el.textContent = '-';
        return;
    }
    
    const formatted = (delta >= 0 ? '+' : '') + delta.toFixed(3);
    el.textContent = formatted;
    el.className = 'delta';
    if (Math.abs(delta) < 0.001) {
        el.classList.add('neutral');
    } else if (delta > 0) {
        el.classList.add('negative');
    } else {
        el.classList.add('positive');
    }
}

// Update comparison select dropdowns
function updateComparisonSelects() {
    const trends = integrityData.trends || [];
    const baselineSelect = /** @type {HTMLSelectElement} */ (document.getElementById('compareBaselineSelect'));
    const targetSelect = /** @type {HTMLSelectElement} */ (document.getElementById('compareTargetSelect'));
    
    if (!baselineSelect || !targetSelect) return;
    
    // Clear existing options
    baselineSelect.innerHTML = '<option value="">Select baseline session...</option>';
    targetSelect.innerHTML = '<option value="">Select target session...</option>';
    
    trends.forEach((trend, index) => {
        const timestamp = trend.timestamp || trend.generated_at || `Entry ${index}`;
        const option1 = document.createElement('option');
        option1.value = String(index);
        option1.textContent = timestamp;
        baselineSelect.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = String(index);
        option2.textContent = timestamp;
        targetSelect.appendChild(option2);
    });
}

// Compare two sessions
function compareSessions() {
    const baselineSelect = /** @type {HTMLSelectElement} */ (document.getElementById('compareBaselineSelect'));
    const targetSelect = /** @type {HTMLSelectElement} */ (document.getElementById('compareTargetSelect'));
    const resultsDiv = document.getElementById('comparisonResults');
    
    if (!baselineSelect || !targetSelect || !resultsDiv) return;
    
    const baselineIdx = parseInt(baselineSelect.value);
    const targetIdx = parseInt(targetSelect.value);
    
    if (isNaN(baselineIdx) || isNaN(targetIdx)) {
        resultsDiv.style.display = 'none';
        return;
    }
    
    const trends = integrityData.trends || [];
    const baseline = trends[baselineIdx];
    const target = trends[targetIdx];
    
    if (!baseline || !target) return;
    
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `
        <div class="comparison-diff">
            <div class="diff-item">
                <div class="diff-label">Confidence Avg</div>
                <div class="diff-values">
                    <span class="diff-baseline">${(baseline.confidence?.avg || 0).toFixed(3)}</span>
                    <span class="diff-arrow">→</span>
                    <span class="diff-target">${(target.confidence?.avg || 0).toFixed(3)}</span>
                </div>
            </div>
            <div class="diff-item">
                <div class="diff-label">Unknown Ratio</div>
                <div class="diff-values">
                    <span class="diff-baseline">${(baseline.unknown_ratio || 0).toFixed(3)}</span>
                    <span class="diff-arrow">→</span>
                    <span class="diff-target">${(target.unknown_ratio || 0).toFixed(3)}</span>
                </div>
            </div>
            <div class="diff-item">
                <div class="diff-label">Segments Review</div>
                <div class="diff-values">
                    <span class="diff-baseline">${(baseline.review_signals?.segments_needing_review || 0).toFixed(1)}</span>
                    <span class="diff-arrow">→</span>
                    <span class="diff-target">${(target.review_signals?.segments_needing_review || 0).toFixed(1)}</span>
                </div>
            </div>
            <div class="diff-item">
                <div class="diff-label">Pattern KB Matches</div>
                <div class="diff-values">
                    <span class="diff-baseline">${(baseline.review_signals?.pattern_kb_matches || 0).toFixed(1)}</span>
                    <span class="diff-arrow">→</span>
                    <span class="diff-target">${(target.review_signals?.pattern_kb_matches || 0).toFixed(1)}</span>
                </div>
            </div>
        </div>
    `;
}

// Export integrity report
function exportIntegrityReport() {
    const report = {
        exported_at: new Date().toISOString(),
        thresholds: thresholds,
        signal: integrityData.signal,
        trends: integrityData.trends
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `integrity_report_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Export integrity report as CSV
function exportIntegrityCsv() {
    const trends = integrityData.trends || [];
    const csvRows = [];
    const headers = ['timestamp', 'confidence_avg', 'confidence_median', 'unknown_ratio', 'segments_review', 'pattern_kb_matches', 'segment_count', 'unknown_segment_count'];
    csvRows.push(headers.join(','));
    
    trends.forEach(t => {
        const row = [
            t.timestamp || '',
            t.confidence?.avg || '',
            t.confidence?.median || '',
            t.unknown_ratio || '',
            t.review_signals?.segments_needing_review || '',
            t.review_signals?.pattern_kb_matches || '',
            t.segment_count || '',
            t.unknown_segment_count || ''
        ];
        csvRows.push(row.join(','));
    });
    
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `integrity_trends_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================
// Event Listeners for Integrity Features
// ============================================

// Reload integrity data
document.getElementById('reloadIntegrityBtn')?.addEventListener('click', loadIntegrityData);

// Export integrity report (toggle between JSON and CSV)
let exportFormat = 'json';
document.getElementById('exportIntegrityBtn')?.addEventListener('click', () => {
    if (exportFormat === 'json') {
        exportIntegrityReport();
        exportFormat = 'csv';
    } else {
        exportIntegrityCsv();
        exportFormat = 'json';
    }
});

// Configure thresholds
document.getElementById('configThresholdsBtn')?.addEventListener('click', () => {
    const modal = document.getElementById('thresholdModal');
    if (modal) {
        // Populate current values
        (/** @type {HTMLInputElement} */ (document.getElementById('confDropThreshold'))).value = String(thresholds.confDropThreshold);
        (/** @type {HTMLInputElement} */ (document.getElementById('unknownSpikeThreshold'))).value = String(thresholds.unknownSpikeThreshold);
        (/** @type {HTMLInputElement} */ (document.getElementById('reviewSpikeThreshold'))).value = String(thresholds.reviewSpikeThreshold);
        (/** @type {HTMLInputElement} */ (document.getElementById('baselineWindow'))).value = String(thresholds.baselineWindow);
        (/** @type {HTMLInputElement} */ (document.getElementById('recentWindow'))).value = String(thresholds.recentWindow);
        
        modal.style.display = 'flex';
    }
});

// Close threshold modal
document.getElementById('closeThresholdModal')?.addEventListener('click', () => {
    const modal = document.getElementById('thresholdModal');
    if (modal) modal.style.display = 'none';
});

document.getElementById('cancelThresholdsBtn')?.addEventListener('click', () => {
    const modal = document.getElementById('thresholdModal');
    if (modal) modal.style.display = 'none';
});

// Save thresholds
document.getElementById('saveThresholdsBtn')?.addEventListener('click', () => {
    thresholds.confDropThreshold = parseFloat((/** @type {HTMLInputElement} */ (document.getElementById('confDropThreshold'))).value);
    thresholds.unknownSpikeThreshold = parseFloat((/** @type {HTMLInputElement} */ (document.getElementById('unknownSpikeThreshold'))).value);
    thresholds.reviewSpikeThreshold = parseFloat((/** @type {HTMLInputElement} */ (document.getElementById('reviewSpikeThreshold'))).value);
    thresholds.baselineWindow = parseInt((/** @type {HTMLInputElement} */ (document.getElementById('baselineWindow'))).value);
    thresholds.recentWindow = parseInt((/** @type {HTMLInputElement} */ (document.getElementById('recentWindow'))).value);
    
    const modal = document.getElementById('thresholdModal');
    if (modal) modal.style.display = 'none';
    
    // Reload with new thresholds
    loadIntegrityData();
});

// Compare sessions
document.getElementById('compareBtn')?.addEventListener('click', compareSessions);