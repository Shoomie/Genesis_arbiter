import os
import threading
import time
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import json

class GenesisVisualizer:
    """
    Real-time interactive dashboard for Genesis Arbiter.
    Serves a Matrix-themed Plotly interface via FastAPI.
    """
    def __init__(self, logger, run_id: str, port: int = 8080):
        self.logger = logger
        self.run_id = run_id
        self.port = port
        self.app = FastAPI()
        self.server_thread = None
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
                "signals": signals
            }

    def start(self):
        """Start the dashboard server in a background thread."""
        def run_server():
            config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="error")
            server = uvicorn.Server(config)
            server.run()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f" [VISUAL] Interactive Matrix Dashboard: http://localhost:{self.port}")

    def _get_html_content(self):
        """Return the Matrix-themed HTML/JS content."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Genesis Arbiter | Command Center</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            background-color: #000000;
            color: #00FF41;
            font-family: 'Courier New', monospace;
            margin: 0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #header {
            padding: 10px 20px;
            border-bottom: 2px solid #003B00;
            background: rgba(0, 20, 0, 0.9);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }
        #title {
            font-size: 1.5em;
            font-weight: bold;
            letter-spacing: 2px;
            text-shadow: 0 0 5px #00FF41;
        }
        #status {
            font-size: 0.9em;
            color: #008F11;
        }
        #plot-container {
            flex-grow: 1;
            width: 100%;
            height: calc(100vh - 60px);
        }
        .matrix-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.05;
            z-index: 0;
        }
        #footer {
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 0.7em;
            color: #003B00;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div id="header">
        <div id="title">GENESIS_ARBITER // LOSS_STREAM</div>
        <div id="status">SYS_STATUS: CONNECTED // RESOLUTION: HI_FIDELITY</div>
    </div>
    
    <div id="plot-container"></div>
    
    <div id="footer">V0.4.2 // RESEARCH_INFRASTRUCTURE</div>

    <script>
        const plotContainer = document.getElementById('plot-container');
        let plotInitialized = false;

        async function updatePlot() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                const history = data.history;
                const signals = data.signals;
                
                if (history.length === 0) return;

                const steps = history.map(h => h.step);
                const losses = history.map(h => h.loss);
                const lrs = history.map(h => h.learning_rate);
                
                // Construct annotations for grokking signals
                const annotations = signals.map(s => ({
                    x: s.step,
                    y: 1.0,
                    xref: 'x',
                    yref: 'paper',
                    text: s.signal_type === 'wwm_activation' ? 'WWM' : 'SPAN',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 0,
                    ay: -40,
                    font: { color: '#00FF41', size: 10 },
                    arrowcolor: '#00FF41',
                    bgcolor: 'rgba(0, 40, 0, 0.8)',
                    bordercolor: '#00FF41',
                    borderwidth: 1
                }));

                const shapes = signals.map(s => ({
                    type: 'line',
                    x0: s.step,
                    y0: 0,
                    x1: s.step,
                    y1: 1,
                    xref: 'x',
                    yref: 'paper',
                    line: {
                        color: 'rgba(0, 255, 65, 0.3)',
                        width: 1,
                        dash: 'dot'
                    }
                }));

                const lossTrace = {
                    x: steps,
                    y: losses,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Loss',
                    line: {
                        color: '#00FF41',
                        width: 2,
                        shape: 'hv' // Step-wise for byte-level precision
                    },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 59, 0, 0.2)',
                    hovertemplate: 'Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                };

                const lrTrace = {
                    x: steps,
                    y: lrs,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'LR',
                    yaxis: 'y2',
                    line: {
                        color: '#008F11',
                        width: 1,
                        dash: 'dash'
                    },
                    hovertemplate: 'Step: %{x}<br>LR: %{y:.2e}<extra></extra>'
                };

                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 40, b: 60, l: 60, r: 60 },
                    showlegend: true,
                    legend: {
                        font: { color: '#008F11', size: 12 },
                        bgcolor: 'rgba(0,0,0,0.5)',
                        x: 1,
                        xanchor: 'right',
                        y: 1
                    },
                    xaxis: {
                        gridcolor: '#001A00',
                        tickfont: { color: '#008F11' },
                        title: { text: 'TRAINING_STEPS', font: { color: '#008F11' } },
                        rangeslider: { visible: false }
                    },
                    yaxis: {
                        gridcolor: '#001A00',
                        tickfont: { color: '#00FF41' },
                        title: { text: 'COMPUTATIONAL_LOSS', font: { color: '#00FF41' } },
                        zeroline: false
                    },
                    yaxis2: {
                        overlaying: 'y',
                        side: 'right',
                        showgrid: false,
                        tickfont: { color: '#008F11' },
                        title: { text: 'LEARNING_RATE', font: { color: '#008F11' } },
                        type: 'log',
                        zeroline: false
                    },
                    hovermode: 'x unified',
                    hoverlabel: {
                        bgcolor: '#000000',
                        bordercolor: '#00FF41',
                        font: { color: '#00FF41', family: 'Courier New' }
                    },
                    annotations: annotations,
                    shapes: shapes,
                    autosize: true
                };

                const config = {
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['select2d', 'lasso2d']
                };

                if (!plotInitialized) {
                    Plotly.newPlot(plotContainer, [lossTrace, lrTrace], layout, config);
                    plotInitialized = true;
                } else {
                    Plotly.react(plotContainer, [lossTrace, lrTrace], layout, config);
                }
            } catch (error) {
                console.error('Data sync failed:', error);
                document.getElementById('status').innerText = 'SYS_STATUS: SYNC_ERROR // RETRYING...';
            }
        }

        // Initial update and periodic polling
        updatePlot();
        setInterval(updatePlot, 2000);

        // Dynamical Fidelity (Resize handler)
        window.addEventListener('resize', () => {
            if (plotInitialized) {
                Plotly.Plots.resize(plotContainer);
            }
        });
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
