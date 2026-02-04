import os
import threading
import time
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import json
from datetime import datetime
from pathlib import Path

class GenesisVisualizer:
    """
    Real-time interactive dashboard for Genesis Arbiter.
    Serves a high-fidelity Matrix-themed interface via FastAPI.
    """
    def __init__(self, logger, run_id: str, port: int = 8080):
        self.logger = logger
        self.run_id = run_id
        self.port = port
        self.app = FastAPI()
        self.server_thread = None
        
        # Real-time state
        self.status = {
            "step": 0,
            "max_steps": 0,
            "loss": 0.0,
            "loss_ema": 0.0,
            "lr": 0.0,
            "tps": 0,
            "eta": "N/A",
            "gpu": "N/A",
            "vram_gb": 0.0,
            "precision": "N/A",
            "wwm_imp": 0.0,
            "span_imp": 0.0,
            "phase": "PH1 (BYTE)",
            "languages": {}
        }
        self.metadata = {
            "run_id": run_id,
            "layers": 0,
            "dim": 0,
            "heads": 0,
            "params": "0M",
            "vocab": 0
        }
        
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_index():
            return self._get_html_content()

        @self.app.get("/api/data")
        async def get_data():
            history = self.logger.get_run_history(self.run_id)
            signals = self.logger.get_grokking_signals(self.run_id)
            return {
                "history": history,
                "signals": signals,
                "status": self.status,
                "metadata": self.metadata
            }

    def update_status(self, metrics: Dict[str, Any]):
        """Update the real-time status buffer."""
        self.status.update(metrics)

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set static model/system metadata."""
        self.metadata.update(metadata)

    def save_telemetry_snapshot(self) -> str:
        """Save the current telemetry and metadata to a JSON file."""
        snapshot = {
            "status": self.status,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        log_dir = Path("logs/telemetry")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = log_dir / f"snapshot_{self.run_id}_{int(time.time())}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(snapshot, f, indent=4)
            return str(file_path)
        except Exception as e:
            print(f" [VISUAL] Failed to save telemetry snapshot: {e}")
            return ""

    def start(self):
        """Start the dashboard server in a background thread."""
        def run_server():
            try:
                config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="error")
                server = uvicorn.Server(config)
                server.run()
            except Exception as e:
                print(f" [VISUAL] Dashboard server failed to start: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f" [VISUAL] Premium Browser Dashboard: http://localhost:{self.port}")

    def _get_html_content(self):
        """Return the premium high-fidelity Matrix-themed dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GENESIS // COMMAND_CENTER</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --matrix-green: #00FF41;
            --matrix-dark: #008F11;
            --matrix-dim: #003B00;
            --bg-black: #000000;
            --glass-bg: rgba(0, 20, 0, 0.85);
            --border-glow: 0 0 10px rgba(0, 255, 65, 0.3);
        }

        * { box-sizing: border-box; }
        body {
            background-color: var(--bg-black);
            color: var(--matrix-green);
            font-family: 'Fira Code', monospace;
            margin: 0;
            overflow-x: hidden;
            background-image: linear-gradient(rgba(0, 255, 65, 0.05) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(0, 255, 65, 0.05) 1px, transparent 1px);
            background-size: 30px 30px;
        }

        /* CRT Scanline Effect */
        body::before {
            content: " ";
            display: block;
            position: fixed;
            top: 0; left: 0; bottom: 0; right: 0;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
                        linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
            z-index: 2000;
            background-size: 100% 4px, 3px 100%;
            pointer-events: none;
        }

        #header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 30px;
            background: var(--glass-bg);
            border-bottom: 2px solid var(--matrix-dim);
            box-shadow: var(--border-glow);
            backdrop-filter: blur(5px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .glitch-text {
            font-weight: 700;
            letter-spacing: 4px;
            text-shadow: 2px 2px var(--matrix-dim);
            animation: flicker 2s infinite;
        }

        @keyframes flicker {
            0% { opacity: 1; }
            5% { opacity: 0.8; }
            10% { opacity: 1; }
            100% { opacity: 1; }
        }

        .container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }

        .card {
            background: var(--glass-bg);
            border: 1px solid var(--matrix-dim);
            border-radius: 4px;
            padding: 15px;
            box-shadow: var(--border-glow);
            transition: all 0.3s ease;
        }

        .card:hover { border-color: var(--matrix-green); }

        .card-title {
            font-size: 0.7em;
            color: var(--matrix-dark);
            margin-bottom: 10px;
            text-transform: uppercase;
            border-bottom: 1px solid var(--matrix-dim);
            padding-bottom: 5px;
            display: flex;
            justify-content: space-between;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            text-shadow: 0 0 10px var(--matrix-green);
        }

        .stat-unit { font-size: 0.4em; color: var(--matrix-dark); margin-left: 5px; }

        .full-width { grid-column: span 4; }
        .span-2 { grid-column: span 2; }

        #plot-container { height: 500px; width: 100%; }

        .lang-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
        }

        .lang-card {
            background: rgba(0, 40, 0, 0.4);
            padding: 8px;
            border-left: 2px solid var(--matrix-green);
            font-size: 0.9em;
        }

        .lang-name { color: var(--matrix-dark); font-size: 0.8em; }

        .progress-bar-bg {
            height: 10px;
            background: var(--matrix-dim);
            margin-top: 15px;
            border-radius: 5px;
            overflow: hidden;
        }

        #progress-fill {
            height: 100%;
            background: var(--matrix-green);
            width: 0%;
            box-shadow: 0 0 15px var(--matrix-green);
            transition: width 0.5s ease;
        }

        @media (max-width: 1200px) {
            .container { grid-template-columns: repeat(2, 1fr); }
            .span-2, .full-width { grid-column: span 2; }
        }

        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
            .span-2, .full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div id="header">
        <div class="glitch-text">GENESIS_ARBITER // COMMAND_CENTER</div>
        <div id="connection-status" style="color: #008F11; font-size: 0.8em;">[ STATUS: INITIALIZING... ]</div>
    </div>

    <div class="container">
        <!-- Main Stats -->
        <div class="card">
            <div class="card-title"><span>TRAINING_PROGRESS</span><span id="pct-text">0%</span></div>
            <div class="stat-value" id="cur-step">0<span class="stat-unit">/ 0 STEPS</span></div>
            <div class="progress-bar-bg"><div id="progress-fill"></div></div>
        </div>

        <div class="card">
            <div class="card-title"><span>THROUGHPUT</span></div>
            <div class="stat-value" id="cur-tps">0<span class="stat-unit">TOK/S</span></div>
            <div id="cur-eta" style="font-size: 0.8em; color: var(--matrix-dark); margin-top: 10px;">ETA: --:--:--</div>
        </div>

        <div class="card">
            <div class="card-title"><span>COMPUTATIONAL_LOSS</span></div>
            <div class="stat-value" id="cur-loss">0.0000</div>
            <div id="cur-ema" style="font-size: 0.8em; color: var(--matrix-dark); margin-top: 10px;">EMA: 0.0000</div>
        </div>

        <div class="card">
            <div class="card-title"><span>RESEARCH_STATE</span></div>
            <div class="stat-value" id="cur-phase" style="font-size: 1.2em; margin-top: 8px;">PH1 (BYTE)</div>
            <div id="cur-imp" style="font-size: 0.8em; color: var(--matrix-dark); margin-top: 10px;">IMPROV: 0.00%</div>
        </div>

        <!-- Plot -->
        <div class="card full-width">
            <div class="card-title"><span>REAL_TIME_LOSS_STREAM</span></div>
            <div id="plot-container"></div>
        </div>

        <!-- Metadata Panels -->
        <div class="card span-2">
            <div class="card-title"><span>MODEL_TOPOLOGY</span></div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                <div>LAYERS: <span id="meta-layers">--</span></div>
                <div>DIM: <span id="meta-dim">--</span></div>
                <div>HEADS: <span id="meta-heads">--</span></div>
            </div>
            <div style="margin-top: 15px; font-size: 0.9em;">PARAMS: <span id="meta-params" style="color: var(--matrix-green);">--</span></div>
            <div style="margin-top: 5px; font-size: 0.9em;">VOCAB: <span id="meta-vocab">--</span></div>
        </div>

        <div class="card span-2">
            <div class="card-title"><span>SYSTEM_RUNTIME</span></div>
            <div id="meta-gpu" style="font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">GPU: --</div>
            <div style="margin-top: 15px; font-size: 0.9em; display: flex; justify-content: space-between;">
                <div>VRAM: <span id="cur-vram">--</span> GB</div>
                <div>PRECISION: <span id="cur-prec">--</span></div>
            </div>
            <div style="margin-top: 5px; font-size: 0.8em; color: #003B00;">RUN_ID: <span id="meta-runid">--</span></div>
        </div>

        <!-- Language Stats -->
        <div class="card full-width">
            <div class="card-title"><span>MULTILINGUAL_TELEMETRY</span></div>
            <div class="lang-grid" id="lang-grid">
                <!-- Lang cards injected here -->
            </div>
        </div>
    </div>

    <script>
        const plotContainer = document.getElementById('plot-container');
        let plotInitialized = false;

        async function updateDashboard() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                // 1. Update Real-time Status
                const s = data.status;
                document.getElementById('cur-step').innerHTML = `${s.step}<span class="stat-unit">/ ${s.max_steps} STEPS</span>`;
                document.getElementById('cur-loss').innerText = s.loss.toFixed(4);
                document.getElementById('cur-ema').innerText = `EMA: ${s.loss_ema.toFixed(4)}`;
                document.getElementById('cur-tps').innerHTML = `${s.tps}<span class="stat-unit">TOK/S</span>`;
                document.getElementById('cur-eta').innerText = `ETA: ${s.eta}`;
                document.getElementById('cur-phase').innerText = s.phase;
                document.getElementById('cur-vram').innerText = s.vram_gb.toFixed(2);
                document.getElementById('cur-prec').innerText = s.precision;
                
                const progress = (s.step / s.max_steps) * 100 || 0;
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('pct-text').innerText = progress.toFixed(1) + '%';
                
                const imp = s.phase.includes("PH3") ? s.span_imp : s.wwm_imp;
                document.getElementById('cur-imp').innerText = `IMPROV: ${imp.toFixed(2)}%`;

                // 2. Update Metadata
                const m = data.metadata;
                document.getElementById('meta-layers').innerText = m.layers;
                document.getElementById('meta-dim').innerText = m.dim;
                document.getElementById('meta-heads').innerText = m.heads;
                document.getElementById('meta-params').innerText = m.params;
                document.getElementById('meta-vocab').innerText = m.vocab;
                document.getElementById('meta-gpu').innerText = `GPU: ${s.gpu}`;
                document.getElementById('meta-runid').innerText = m.run_id;

                // 3. Update Languages
                const langGrid = document.getElementById('lang-grid');
                langGrid.innerHTML = '';
                
                const sortedLangs = Object.entries(s.languages)
                    .filter(([_, stats]) => stats.ema !== null)
                    .sort(([, a], [, b]) => b.ema - a.ema);

                for (const [lang, stats] of sortedLangs) {
                    const div = document.createElement('div');
                    div.className = 'lang-card';
                    div.innerHTML = `<div class="lang-name">${lang.toUpperCase()}</div><div>${stats.ema.toFixed(4)}</div>`;
                    langGrid.appendChild(div);
                }

                // 4. Update Plot
                const history = data.history;
                const signals = data.signals;
                if (history.length > 0) {
                    const steps = history.map(h => h.step);
                    const losses = history.map(h => h.loss);
                    const lrs = history.map(h => h.learning_rate);
                    
                    const annotations = signals.map(sig => ({
                        x: sig.step, y: 1.0, xref: 'x', yref: 'paper',
                        text: sig.signal_type === 'wwm_activation' ? 'WWM_LOCK' : 'SPAN_GEN',
                        showarrow: true, arrowhead: 2, ax: 0, ay: -30,
                        font: { color: '#00FF41', size: 9 },
                        arrowcolor: '#00FF41', bgcolor: 'rgba(0, 40, 0, 0.9)', bordercolor: '#00FF41'
                    }));

                    const lossTrace = {
                        x: steps, y: losses, type: 'scatter', mode: 'lines', name: 'LOSS',
                        line: { color: '#00FF41', width: 2, shape: 'hv' },
                        fill: 'tozeroy', fillcolor: 'rgba(0, 59, 0, 0.15)'
                    };

                    const lrTrace = {
                        x: steps, y: lrs, type: 'scatter', mode: 'lines', name: 'LR',
                        yaxis: 'y2', line: { color: '#008F11', width: 1, dash: 'dot' }
                    };

                    // Calculate dynamic ranges for auto-centering while preserving user zoom (span)
                    let lossRange = null;
                    let lrRange = null;

                    if (plotInitialized && plotContainer.layout) {
                        // LOSS (Linear)
                        const currentLoss = s.loss;
                        const y = plotContainer.layout.yaxis;
                        if (y && y.range) {
                            const span = y.range[1] - y.range[0];
                            lossRange = [currentLoss - span/2, currentLoss + span/2];
                        } else {
                            // Default view: +/- 20% of value
                            lossRange = [currentLoss * 0.8, currentLoss * 1.2];
                        }

                        // LR (Log)
                        const currentLr = s.lr;
                        const y2 = plotContainer.layout.yaxis2;
                        if (y2 && y2.range) {
                            const span = y2.range[1] - y2.range[0];
                            const curLog = Math.log10(currentLr);
                            lrRange = [curLog - span/2, curLog + span/2];
                        }
                        // Default for LR handled by autorange if null
                    }

                    const layout = {
                        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: { t: 30, b: 40, l: 50, r: 50 },
                        showlegend: true, legend: { font: { color: '#008F11' }, bgcolor: 'rgba(0,0,0,0.5)', x: 1, xanchor: 'right' },
                        
                        // Interaction: Zoom (Scaling) preferred over Pan
                        dragmode: 'zoom', 
                        
                        // X-AXIS: Locked (Horizontal length intact)
                        xaxis: { 
                            gridcolor: '#001A00', tickfont: { color: '#008F11' }, zeroline: false,
                            fixedrange: true, // Prevent horizontal zoom
                            autorange: true   // Always show full history
                        },
                        
                        // Y-AXIS (Loss): Vertical Zoom Only + Auto-Center
                        yaxis: { 
                            gridcolor: '#001A00', tickfont: { color: '#00FF41' }, zeroline: false,
                            fixedrange: false, // Allow vertical zoom
                            range: lossRange   // Force center
                        },
                        
                        // Y-AXIS 2 (LR): Vertical Zoom Only + Auto-Center
                        yaxis2: { 
                            overlaying: 'y', side: 'right', showgrid: false, tickfont: { color: '#008F11' }, type: 'log',
                            fixedrange: false, // Allow vertical zoom
                            range: lrRange     // Force center
                        },
                        
                        hovermode: 'x unified', annotations: annotations, autosize: true
                    };

                    const config = {
                        responsive: true, 
                        displaylogo: false, 
                        scrollZoom: true, // Enable scroll wheel scaling
                        displayModeBar: false // Clean look
                    };

                    if (!plotInitialized) {
                        Plotly.newPlot(plotContainer, [lossTrace, lrTrace], layout, config);
                        plotInitialized = true;
                    } else {
                        Plotly.react(plotContainer, [lossTrace, lrTrace], layout, config);
                    }
                }

                document.getElementById('connection-status').innerText = '[ STATUS: ONLINE // SYNC_OK ]';
                document.getElementById('connection-status').style.color = '#00FF41';

            } catch (err) {
                console.error(err);
                document.getElementById('connection-status').innerText = '[ STATUS: OFFLINE // RETRYING... ]';
                document.getElementById('connection-status').style.color = '#FF0000';
            }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
        window.addEventListener('resize', () => plotInitialized && Plotly.Plots.resize(plotContainer));
    </script>
</body>
</html>
        """

if __name__ == "__main__":
    # Test stub
    class MockLogger:
        def get_run_history(self, rid): return [{"step": i, "loss": 5.0 - i/1000, "learning_rate": 1e-4} for i in range(100)]
        def get_grokking_signals(self, rid): return [{"step": 50, "signal_type": "wwm_activation"}]
    
    vis = GenesisVisualizer(MockLogger(), "test_run")
    vis.start()
    while True: time.sleep(1)
