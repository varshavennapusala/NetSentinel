/**
 * NetSentinel — Dynamic Multi-Algorithm Inference Suite
 * Client Logic, Chart.js Visualizations, and Comparative Analysis
 */

let appState = {
    currentDataset: null,
    presets: {},
    benchmarks: null,
    currentResults: null,
    activeChartTab: 'multimetric',
    chartInstance: null
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const pRes = await fetch('/api/presets');
        if (pRes.ok) {
            appState.presets = await pRes.json();
        }
    } catch (err) {
        console.error('Failed to load presets:', err);
    }

    try {
        const bRes = await fetch('/api/benchmarks');
        if (bRes.ok) {
            appState.benchmarks = await bRes.json();
        }
    } catch (err) {
        console.error('Failed to load benchmarks:', err);
    }
});

/**
 * Switch from View 1 (Dataset Selection) to View 2 (Input & Comparative Workspace)
 */
function selectDataset(track) {
    appState.currentDataset = track;
    appState.currentResults = null;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    const selectionView = document.getElementById('datasetSelectionView');
    const workspaceView = document.getElementById('workspaceView');
    const resultsSection = document.getElementById('resultsSection');

    if (selectionView) selectionView.style.display = 'none';
    if (workspaceView) workspaceView.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';

    // Toggle specific form
    const formReg = document.getElementById('formRegression');
    const formClf = document.getElementById('formClassification');
    const formClust = document.getElementById('formClustering');

    if (formReg) formReg.style.display = (track === 'regression') ? 'block' : 'none';
    if (formClf) formClf.style.display = (track === 'classification') ? 'block' : 'none';
    if (formClust) formClust.style.display = (track === 'clustering') ? 'block' : 'none';

    // Update track switcher tabs
    ['regression', 'classification', 'clustering'].forEach(t => {
        const tabBtn = document.getElementById(`trackTab_${t}`);
        if (tabBtn) {
            if (t === track) {
                tabBtn.classList.add('active');
            } else {
                tabBtn.classList.remove('active');
            }
        }
    });

    // Update active dataset headers & pills
    const pill = document.getElementById('activeDatasetPill');
    const title = document.getElementById('activeDatasetTitle');
    const desc = document.getElementById('activeDatasetDesc');

    if (track === 'regression') {
        if (pill) pill.textContent = 'Selected: Track 1 · UNSW-NB15 QoS Degradation (Regression)';
        if (title) title.textContent = 'UNSW-NB15 Flow QoS Packet Loss Estimation';
        if (desc) desc.textContent = 'Configure bidirectional flow telemetry parameters to evaluate predictions across all 10 trained regression algorithms.';
    } else if (track === 'classification') {
        if (pill) pill.textContent = 'Selected: Track 2 · CIC-IDS2017 Intrusion Detection (Classification)';
        if (title) title.textContent = 'CIC-IDS2017 Network Threat Detection & Triage';
        if (desc) desc.textContent = 'Enter network flow statistics and TCP flags to evaluate multi-class threat classifications across all 10 trained classification algorithms.';
    } else if (track === 'clustering') {
        if (pill) pill.textContent = 'Selected: Track 3 · CIC-IDS2017 Traffic Behavioral Profiling (Clustering)';
        if (title) title.textContent = 'CIC-IDS2017 Unsupervised Behavioral Flow Profiling';
        if (desc) desc.textContent = 'Input network flow telemetry to assign traffic behavioral clusters across all 5 clustering configurations.';
    }

    // Render quick preset chips
    renderPresetChips(track);

    // Update chart tab labels for track
    updateChartTabLabels(track);
    appState.activeChartTab = 'multimetric';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Return from Workspace back to the 3-Dataset Selection screen
 */
function returnToDatasetSelection() {
    appState.currentDataset = null;
    appState.currentResults = null;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    document.getElementById('workspaceView').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('datasetSelectionView').style.display = 'block';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Render Quick Presets Chips
 */
function renderPresetChips(track) {
    const container = document.getElementById('presetsList');
    if (!container) return;
    container.innerHTML = '';

    const trackPresets = appState.presets[track] || {};
    const keys = Object.keys(trackPresets);

    if (keys.length === 0) {
        container.innerHTML = '<span class="field-hint">No presets available</span>';
        return;
    }

    keys.forEach((key, index) => {
        const p = trackPresets[key];
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'preset-chip';
        chip.textContent = p.label;
        chip.title = p.desc || '';
        chip.onclick = () => applyPreset(track, key);
        container.appendChild(chip);

        // Auto-apply first preset on initial track selection
        if (index === 0 && !appState.currentResults) {
            applyPreset(track, key);
        }
    });
}

/**
 * Apply Preset Values to Form
 */
function applyPreset(track, presetKey) {
    const preset = appState.presets[track]?.[presetKey];
    if (!preset || !preset.data) return;

    const data = preset.data;

    if (track === 'regression') {
        if ('sbytes' in data) document.getElementById('reg_sbytes').value = data.sbytes;
        if ('dbytes' in data) document.getElementById('reg_dbytes').value = data.dbytes;
        if ('spkts' in data) document.getElementById('reg_spkts').value = data.spkts;
        if ('dpkts' in data) document.getElementById('reg_dpkts').value = data.dpkts;
        if ('dur' in data) document.getElementById('reg_dur').value = data.dur;
        if ('sload' in data) document.getElementById('reg_sload').value = data.sload;
        if ('dload' in data) document.getElementById('reg_dload').value = data.dload;
        if ('rate' in data) document.getElementById('reg_rate').value = data.rate;
        if ('sttl' in data) document.getElementById('reg_sttl').value = data.sttl;
        if ('dttl' in data) document.getElementById('reg_dttl').value = data.dttl;
    } else if (track === 'classification') {
        if ('port' in data) document.getElementById('clf_port').value = data.port;
        if ('dur' in data) document.getElementById('clf_dur').value = data.dur;
        if ('fwd_pkts' in data) document.getElementById('clf_fwd_pkts').value = data.fwd_pkts;
        if ('bwd_pkts' in data) document.getElementById('clf_bwd_pkts').value = data.bwd_pkts;
        if ('tot_fwd_len' in data) document.getElementById('clf_tot_fwd_len').value = data.tot_fwd_len;
        if ('fwd_len_mean' in data) document.getElementById('clf_fwd_len_mean').value = data.fwd_len_mean;
        if ('bytes_s' in data) document.getElementById('clf_bytes_s').value = data.bytes_s;
        if ('pkts_s' in data) document.getElementById('clf_pkts_s').value = data.pkts_s;
        if ('iat_mean' in data) document.getElementById('clf_iat_mean').value = data.iat_mean;
        if ('pkt_len_mean' in data) document.getElementById('clf_pkt_len_mean').value = data.pkt_len_mean;
        if ('pkt_len_std' in data) document.getElementById('clf_pkt_len_std').value = data.pkt_len_std;
        if ('syn_flag' in data) document.getElementById('clf_syn_flag').value = data.syn_flag;
        if ('ack_flag' in data) document.getElementById('clf_ack_flag').value = data.ack_flag;
    } else if (track === 'clustering') {
        if ('port' in data) document.getElementById('cl_port').value = data.port;
        if ('dur' in data) document.getElementById('cl_dur').value = data.dur;
        if ('fwd_pkts' in data) document.getElementById('cl_fwd_pkts').value = data.fwd_pkts;
        if ('bwd_pkts' in data) document.getElementById('cl_bwd_pkts').value = data.bwd_pkts;
        if ('tot_fwd_len' in data) document.getElementById('cl_tot_fwd_len').value = data.tot_fwd_len;
        if ('fwd_len_mean' in data) document.getElementById('cl_fwd_len_mean').value = data.fwd_len_mean;
        if ('bytes_s' in data) document.getElementById('cl_bytes_s').value = data.bytes_s;
        if ('pkts_s' in data) document.getElementById('cl_pkts_s').value = data.pkts_s;
        if ('iat_mean' in data) document.getElementById('cl_iat_mean').value = data.iat_mean;
        if ('pkt_len_mean' in data) document.getElementById('cl_pkt_len_mean').value = data.pkt_len_mean;
        if ('pkt_len_std' in data) document.getElementById('cl_pkt_len_std').value = data.pkt_len_std;
        if ('syn_flag' in data) document.getElementById('cl_syn_flag').value = data.syn_flag;
        if ('ack_flag' in data) document.getElementById('cl_ack_flag').value = data.ack_flag;
    }
}

/**
 * Handle Form Submission for any track
 */
async function handleFormSubmit(event, track) {
    event.preventDefault();

    let submitBtn;
    let url;
    let payload = {};

    if (track === 'regression') {
        submitBtn = document.getElementById('submitBtnRegression');
        url = '/api/predict/regression';
        payload = {
            sbytes: parseFloat(document.getElementById('reg_sbytes').value),
            dbytes: parseFloat(document.getElementById('reg_dbytes').value),
            spkts: parseInt(document.getElementById('reg_spkts').value),
            dpkts: parseInt(document.getElementById('reg_dpkts').value),
            dur: parseFloat(document.getElementById('reg_dur').value),
            sload: parseFloat(document.getElementById('reg_sload').value),
            dload: parseFloat(document.getElementById('reg_dload').value),
            rate: parseFloat(document.getElementById('reg_rate').value),
            sttl: parseInt(document.getElementById('reg_sttl').value),
            dttl: parseInt(document.getElementById('reg_dttl').value)
        };
    } else if (track === 'classification') {
        submitBtn = document.getElementById('submitBtnClassification');
        url = '/api/predict/classification';
        payload = {
            port: parseInt(document.getElementById('clf_port').value),
            dur: parseInt(document.getElementById('clf_dur').value),
            fwd_pkts: parseInt(document.getElementById('clf_fwd_pkts').value),
            bwd_pkts: parseInt(document.getElementById('clf_bwd_pkts').value),
            tot_fwd_len: parseFloat(document.getElementById('clf_tot_fwd_len').value),
            fwd_len_mean: parseFloat(document.getElementById('clf_fwd_len_mean').value),
            bytes_s: parseFloat(document.getElementById('clf_bytes_s').value),
            pkts_s: parseFloat(document.getElementById('clf_pkts_s').value),
            iat_mean: parseFloat(document.getElementById('clf_iat_mean').value),
            pkt_len_mean: parseFloat(document.getElementById('clf_pkt_len_mean').value),
            pkt_len_std: parseFloat(document.getElementById('clf_pkt_len_std').value),
            syn_flag: parseInt(document.getElementById('clf_syn_flag').value),
            ack_flag: parseInt(document.getElementById('clf_ack_flag').value)
        };
    } else if (track === 'clustering') {
        submitBtn = document.getElementById('submitBtnClustering');
        url = '/api/predict/clustering';
        payload = {
            port: parseInt(document.getElementById('cl_port').value),
            dur: parseFloat(document.getElementById('cl_dur').value),
            fwd_pkts: parseInt(document.getElementById('cl_fwd_pkts').value),
            bwd_pkts: parseInt(document.getElementById('cl_bwd_pkts').value),
            tot_fwd_len: parseFloat(document.getElementById('cl_tot_fwd_len').value),
            fwd_len_mean: parseFloat(document.getElementById('cl_fwd_len_mean').value),
            bytes_s: parseFloat(document.getElementById('cl_bytes_s').value),
            pkts_s: parseFloat(document.getElementById('cl_pkts_s').value),
            iat_mean: parseFloat(document.getElementById('cl_iat_mean').value),
            pkt_len_mean: parseFloat(document.getElementById('cl_pkt_len_mean').value),
            pkt_len_std: parseFloat(document.getElementById('cl_pkt_len_std').value),
            syn_flag: parseInt(document.getElementById('cl_syn_flag').value),
            ack_flag: parseInt(document.getElementById('cl_ack_flag').value)
        };
    }

    const originalBtnText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Running All Algorithms...';

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!data.success) {
            alert('Error running models: ' + (data.error || 'Unknown error'));
            return;
        }

        appState.currentResults = data;
        displayResults(data, track);

    } catch (err) {
        console.error('Inference error:', err);
        alert('Failed to connect to the NetSentinel prediction server.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
}

/**
 * Update Chart Tab Labels based on selected track
 */
function updateChartTabLabels(track) {
    const tab1 = document.getElementById('tabChart1');
    const tab2 = document.getElementById('tabChart2');
    const tab3 = document.getElementById('tabChart3');
    if (!tab1 || !tab2 || !tab3) return;

    if (track === 'regression') {
        tab1.textContent = 'R² & Fit Diagnostics (Cell 58)';
        tab2.textContent = 'Error Metric Analysis (RMSE / MAE)';
        tab3.textContent = 'Live Packet Loss Predictions';
    } else if (track === 'classification') {
        tab1.textContent = 'Multi-Metric Comparison (Cell 68)';
        tab2.textContent = 'Consolidated ROC-AUC & F1';
        tab3.textContent = 'Live Model Verdict Distribution';
    } else if (track === 'clustering') {
        tab1.textContent = 'Silhouette & Davies-Bouldin (Cell 22)';
        tab2.textContent = 'Calinski-Harabasz Variance Ratio';
        tab3.textContent = 'Centroid Proximity & Cluster Distance';
    }
}

/**
 * Display outputs, champion banner, comparison table, and interactive charts
 */
function displayResults(data, track) {
    appState.currentResults = data;
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    const champion = data.champion;

    // 1. Champion Highlight Banner
    const modelNameEl = document.getElementById('championModelName');
    const answerValEl = document.getElementById('championAnswerValue');
    const answerUnitEl = document.getElementById('championAnswerUnit');
    const tierBadgeEl = document.getElementById('championTierBadge');
    const recEl = document.getElementById('championRecommendation');

    modelNameEl.textContent = champion.model_name;

    if (track === 'regression') {
        answerValEl.textContent = `${champion.predicted_value.toFixed(2)} pkts`;
        answerUnitEl.textContent = `Source Packet Loss (sloss) · Consensus: ${champion.consensus_mean} pkts`;
        tierBadgeEl.textContent = champion.tier_badge;
        
        if (champion.color_theme === 'success') {
            tierBadgeEl.style.backgroundColor = '#ecfdf5';
            tierBadgeEl.style.color = '#065f46';
            tierBadgeEl.style.borderColor = '#a7f3d0';
        } else if (champion.color_theme === 'warning') {
            tierBadgeEl.style.backgroundColor = '#fffbeb';
            tierBadgeEl.style.color = '#92400e';
            tierBadgeEl.style.borderColor = '#fde68a';
        } else {
            tierBadgeEl.style.backgroundColor = '#fef2f2';
            tierBadgeEl.style.color = '#991b1b';
            tierBadgeEl.style.borderColor = '#fecaca';
        }
        recEl.textContent = champion.recommendation;

    } else if (track === 'classification') {
        answerValEl.textContent = champion.verdict;
        answerUnitEl.textContent = `Security Verdict · Agreement: ${champion.consensus_agreement}`;
        tierBadgeEl.textContent = champion.tier_badge;
        
        if (!champion.is_attack) {
            tierBadgeEl.style.backgroundColor = '#ecfdf5';
            tierBadgeEl.style.color = '#065f46';
            tierBadgeEl.style.borderColor = '#a7f3d0';
        } else {
            tierBadgeEl.style.backgroundColor = '#fef2f2';
            tierBadgeEl.style.color = '#991b1b';
            tierBadgeEl.style.borderColor = '#fecaca';
        }
        recEl.textContent = champion.security_advice;

    } else if (track === 'clustering') {
        answerValEl.textContent = champion.persona_name;
        answerUnitEl.textContent = `Behavioral Tag: ${champion.persona_badge} · Centroid Distance: ${champion.distance_to_center}`;
        tierBadgeEl.textContent = `Assigned Segment: Cluster #${champion.cluster_id}`;
        tierBadgeEl.style.backgroundColor = '#eff6ff';
        tierBadgeEl.style.color = '#1e40af';
        tierBadgeEl.style.borderColor = '#bfdbfe';
        recEl.textContent = `${champion.persona_desc} Recommended Traffic Strategy: ${champion.network_strategy}`;
    }

    // 2. Render Full Comparison Table directly reflecting notebook tables
    renderComparisonTable(data.all_algorithms, track);

    // 3. Update Chart Tabs and Render Chart
    updateChartTabLabels(track);
    renderChart(appState.activeChartTab || 'multimetric');

    // Smooth scroll down to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Render the full comparison table across all algorithms with complete notebook rubric metrics
 */
function renderComparisonTable(algorithms, track) {
    const thead = document.getElementById('algorithmsTableHead');
    const tbody = document.getElementById('algorithmsTableBody');
    const caption = document.getElementById('tableCaption');
    const title = document.getElementById('tableTitle');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (track === 'regression') {
        if (title) title.textContent = 'Consolidated 10-Algorithm Regression Comparison Table (Notebook Cell 57)';
        if (caption) caption.textContent = 'All 10 regression algorithms evaluated with live predictions alongside R², RMSE, and MAE benchmarks';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">Rank</th>
                    <th>Model Architecture</th>
                    <th>Predicted Loss (Live)</th>
                    <th>R² Score (↑)</th>
                    <th>5-Fold CV R²</th>
                    <th>RMSE (↓)</th>
                    <th>MAE (↓)</th>
                    <th>Evaluation Status</th>
                </tr>
            `;
        }
        algorithms.forEach(item => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${item.rank}</span></td>
                <td><strong>${item.model}</strong></td>
                <td><span class="table-num"><strong>${item.prediction_label}</strong></span></td>
                <td><span class="table-num font-mono"><strong>${item.r2 !== undefined ? item.r2.toFixed(4) : '-'}</strong></span></td>
                <td><span class="table-num font-mono">${item.cv_r2 !== undefined ? item.cv_r2.toFixed(4) : '-'}</span></td>
                <td><span class="table-num font-mono">${item.rmse !== undefined ? item.rmse.toFixed(2) : '-'}</span></td>
                <td><span class="table-num font-mono">${item.mae !== undefined ? item.mae.toFixed(2) : '-'}</span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });

    } else if (track === 'classification') {
        if (title) title.textContent = 'Consolidated 10-Algorithm Classification Comparison Table (Notebook Cell 68)';
        if (caption) caption.textContent = 'All 10 classification algorithms evaluated with live traffic predictions alongside Accuracy, Precision, Recall, F1, and ROC-AUC';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">Rank</th>
                    <th>Model Architecture</th>
                    <th>Model Prediction (Live)</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 (weighted)</th>
                    <th>ROC-AUC</th>
                    <th>Evaluation Status</th>
                </tr>
            `;
        }
        algorithms.forEach(item => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            const isBenign = item.prediction === 'BENIGN';
            const callColor = isBenign ? 'style="color:#15803d;font-weight:700;"' : 'style="color:#b91c1c;font-weight:700;"';
            const callBadge = isBenign ? 'badge-green' : 'badge-red';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${item.rank}</span></td>
                <td><strong>${item.model}</strong></td>
                <td><span class="badge ${callBadge}" ${callColor}>${item.prediction_label}</span></td>
                <td><span class="table-num font-mono">${item.acc !== undefined ? (item.acc * 100).toFixed(2) + '%' : '-'}</span></td>
                <td><span class="table-num font-mono">${item.prec !== undefined ? (item.prec * 100).toFixed(2) + '%' : '-'}</span></td>
                <td><span class="table-num font-mono">${item.recall !== undefined ? (item.recall * 100).toFixed(2) + '%' : '-'}</span></td>
                <td><span class="table-num font-mono"><strong>${item.f1 !== undefined ? (item.f1 * 100).toFixed(2) + '%' : '-'}</strong></span></td>
                <td><span class="table-num font-mono"><strong>${item.auc !== undefined ? (item.auc * 100).toFixed(2) + '%' : '-'}</strong></span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });

    } else if (track === 'clustering') {
        if (title) title.textContent = 'Consolidated Clustering Formulations Comparison Table (Notebook Cell 22 & 25)';
        if (caption) caption.textContent = 'Unsupervised partitionings evaluated with Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">Rank</th>
                    <th>Clustering Formulation</th>
                    <th>k</th>
                    <th>Assigned Cluster</th>
                    <th>Traffic Profile</th>
                    <th>Silhouette Score (↑)</th>
                    <th>Davies-Bouldin (↓)</th>
                    <th>Calinski-Harabasz (↑)</th>
                    <th>Status</th>
                </tr>
            `;
        }
        algorithms.forEach(item => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${item.rank}</span></td>
                <td><strong>${item.model}</strong></td>
                <td><span class="table-num font-mono">${item.k}</span></td>
                <td><span class="table-num font-mono"><strong>${item.assigned_cluster}</strong></span></td>
                <td><span class="badge badge-blue">${item.profile_category}</span></td>
                <td><span class="table-num font-mono"><strong>${item.silhouette.toFixed(4)}</strong></span></td>
                <td><span class="table-num font-mono">${item.dbi.toFixed(4)}</span></td>
                <td><span class="table-num font-mono">${item.chi.toFixed(2)}</span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }
}

/**
 * Switch Chart Tabs
 */
function switchChartTab(tabKey) {
    appState.activeChartTab = tabKey;
    const tab1 = document.getElementById('tabChart1');
    const tab2 = document.getElementById('tabChart2');
    const tab3 = document.getElementById('tabChart3');

    if (tab1) tab1.classList.toggle('active', tabKey === 'multimetric');
    if (tab2) tab2.classList.toggle('active', tabKey === 'error' || tabKey === 'roc');
    if (tab3) tab3.classList.toggle('active', tabKey === 'predictions');

    renderChart(tabKey);
}

/**
 * Render Chart.js Canvas based on current track and tab
 */
function renderChart(tabKey) {
    const canvas = document.getElementById('comparisonChartCanvas');
    if (!canvas || !appState.currentResults) return;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    const ctx = canvas.getContext('2d');
    const track = appState.currentDataset;
    const algorithms = appState.currentResults.all_algorithms;

    const labels = algorithms.map(a => a.model.replace(' Regressor', '').replace(' Classifier', ''));

    if (track === 'regression') {
        if (tabKey === 'multimetric') {
            const r2Vals = algorithms.map(a => a.r2);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'R² Score (Explained Variance)',
                        data: r2Vals,
                        backgroundColor: algorithms.map(a => a.is_champion ? '#2563eb' : '#94a3b8'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: { callbacks: { label: ctx => ` R² Score: ${ctx.parsed.y.toFixed(4)}` } }
                    },
                    scales: {
                        y: { min: 0.95, max: 1.0, title: { display: true, text: 'R² Metric (Higher is better)' } }
                    }
                }
            });
        } else if (tabKey === 'error' || tabKey === 'roc') {
            const rmseVals = algorithms.map(a => a.rmse);
            const maeVals = algorithms.map(a => a.mae);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'RMSE (Root Mean Squared Error)', data: rmseVals, backgroundColor: '#f97316', borderRadius: 4 },
                        { label: 'MAE (Mean Absolute Error)', data: maeVals, backgroundColor: '#38bdf8', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'top' } },
                    scales: { y: { title: { display: true, text: 'Error in Packets (Lower is better)' } } }
                }
            });
        } else {
            // Live predictions
            const preds = algorithms.map(a => a.prediction);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Predicted Source Packet Loss (packets)',
                        data: preds,
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { title: { display: true, text: 'Predicted Loss (sloss, packets)' } } }
                }
            });
        }

    } else if (track === 'classification') {
        if (tabKey === 'multimetric') {
            const accVals = algorithms.map(a => a.acc * 100);
            const f1Vals = algorithms.map(a => a.f1 * 100);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Accuracy (%)', data: accVals, backgroundColor: '#2563eb', borderRadius: 4 },
                        { label: 'Weighted F1 Score (%)', data: f1Vals, backgroundColor: '#7c3aed', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { min: 70, max: 100, title: { display: true, text: 'Score Percentage (%)' } } }
                }
            });
        } else if (tabKey === 'error' || tabKey === 'roc') {
            const aucVals = algorithms.map(a => a.auc * 100);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'ROC-AUC OvR (%)',
                        data: aucVals,
                        backgroundColor: algorithms.map(a => a.is_champion ? '#059669' : '#64748b'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { min: 90, max: 100, title: { display: true, text: 'ROC-AUC (%)' } } }
                }
            });
        } else {
            // Live verdict distribution
            const preds = algorithms.map(a => a.prediction);
            const counts = {};
            preds.forEach(p => counts[p] = (counts[p] || 0) + 1);

            appState.chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(counts),
                    datasets: [{
                        data: Object.values(counts),
                        backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' },
                        title: { display: true, text: '10-Algorithm Consensus Distribution on Current Telemetry' }
                    }
                }
            });
        }

    } else if (track === 'clustering') {
        if (tabKey === 'multimetric') {
            const silVals = algorithms.map(a => a.silhouette);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: algorithms.map(a => a.model),
                    datasets: [{
                        label: 'Silhouette Score (Partition Quality)',
                        data: silVals,
                        backgroundColor: '#7c3aed',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { min: 0.15, max: 0.25, title: { display: true, text: 'Silhouette Score' } } }
                }
            });
        } else if (tabKey === 'error' || tabKey === 'roc') {
            const chiVals = algorithms.map(a => a.chi);
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: algorithms.map(a => a.model),
                    datasets: [{
                        label: 'Calinski-Harabasz Index (Variance Ratio)',
                        data: chiVals,
                        backgroundColor: '#0284c7',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { title: { display: true, text: 'Calinski-Harabasz Index (Higher is better)' } } }
                }
            });
        } else {
            // Centroid distances
            const dists = appState.currentResults.champion.centroid_distances || [1.0, 2.0, 3.0];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: dists.map((_, i) => `Cluster ${i} Centroid`),
                    datasets: [{
                        label: 'Euclidean Distance from Submitted Flow Telemetry',
                        data: dists,
                        backgroundColor: ['#10b981', '#3b82f6', '#ef4444'],
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { title: { display: true, text: 'Euclidean Distance to Center (Lower = Stronger Affinity)' } } }
                }
            });
        }
    }
}

/**
 * Scroll to inputs helper
 */
function scrollToInputs() {
    const formCard = document.querySelector('.form-card');
    if (formCard) {
        formCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}
