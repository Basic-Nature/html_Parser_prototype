let allMetrics = [];
let charts = {};

// Load metrics from API
async function loadMetrics() {
    const params = new URLSearchParams();

    const handler = document.getElementById('handlerFilter').value;
    const state = document.getElementById('stateFilter').value;
    const minConf = document.getElementById('minConfidence').value;
    const limit = document.getElementById('limitFilter').value;

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
    document.getElementById('totalCount').textContent = allMetrics.length;

    const confidences = allMetrics
        .map(m => m.quality_metrics?.extraction_confidence)
        .filter(c => c != null);
    const avgConf = confidences.length > 0
        ? (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(3)
        : 'N/A';
    document.getElementById('avgConfidence').textContent = avgConf;

    const rows = allMetrics
        .map(m => m.row_count)
        .filter(r => r != null);
    const avgRows = rows.length > 0
        ? Math.round(rows.reduce((a, b) => a + b, 0) / rows.length)
        : 'N/A';
    document.getElementById('avgRows').textContent = avgRows;

    const cols = allMetrics
        .map(m => m.column_count)
        .filter(c => c != null);
    const avgCols = cols.length > 0
        ? Math.round(cols.reduce((a, b) => a + b, 0) / cols.length)
        : 'N/A';
    document.getElementById('avgCols').textContent = avgCols;
}

// Update charts
function updateCharts() {
    // Confidence over time
    const labels = allMetrics.map(m => m.timestamp.slice(0, 8)); // YYYYMMDD
    const confidences = allMetrics.map(m => m.quality_metrics?.extraction_confidence || 0);

    if (charts.confidence) charts.confidence.destroy();
    charts.confidence = new Chart(document.getElementById('confidenceChart'), {
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
    charts.handler = new Chart(document.getElementById('handlerChart'), {
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
    charts.distribution = new Chart(document.getElementById('distributionChart'), {
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
    charts.emptyRow = new Chart(document.getElementById('emptyRowChart'), {
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
    tbody.innerHTML = '';

    if (allMetrics.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 2rem;">No data</td></tr>';
        return;
    }

    allMetrics.slice(0, 20).forEach(m => {
        const conf = m.quality_metrics?.extraction_confidence;
        let confBadge = '<span class="confidence-badge confidence-unknown">N/A</span>';
        if (conf != null) {
            const confPct = (conf * 100).toFixed(1) + '%';
            if (conf >= 0.8) confBadge = `<span class="confidence-badge confidence-high">${confPct}</span>`;
            else if (conf >= 0.5) confBadge = `<span class="confidence-badge confidence-medium">${confPct}</span>`;
            else confBadge = `<span class="confidence-badge confidence-low">${confPct}</span>`;
        }

        const emptyRatio = m.quality_metrics?.empty_row_ratio;
        const emptyPct = emptyRatio != null ? (emptyRatio * 100).toFixed(1) + '%' : 'N/A';

        const row = `<tr>
            <td>${m.timestamp}</td>
            <td>${m.handler || 'N/A'}</td>
            <td>${m.state || 'N/A'}</td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${m.contest || 'N/A'}</td>
            <td>${m.row_count || 'N/A'}</td>
            <td>${m.column_count || 'N/A'}</td>
            <td>${confBadge}</td>
            <td>${emptyPct}</td>
        </tr>`;
        tbody.innerHTML += row;
    });
}

// Update state filter dropdown
function updateStateFilter() {
    const stateFilter = document.getElementById('stateFilter');
    const currentValue = stateFilter.value;

    const states = new Set(allMetrics.map(m => m.state).filter(s => s));

    stateFilter.innerHTML = '<option value="">All States</option>';
    Array.from(states).sort().forEach(state => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = state;
        if (state === currentValue) option.selected = true;
        stateFilter.appendChild(option);
    });
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
