/* ======================================================================
   Retrieve chart data from app.py
   ======================================================================= */
function getChartJson(scriptId) {
  const el = document.getElementById(scriptId);
  if (!el) {
    console.error(`[DX] Missing <script id="${scriptId}">`);
    return null;
  }
  try {
    return JSON.parse(el.textContent);
  } catch (err) {
    console.error(`[DX] JSON.parse failed for ${scriptId}:`, err);
    return null;
  }
}

/* ======================================================================
   Render Plotly charts
   ======================================================================= */

function renderPlot(containerId, scriptId) {
  const fig = getChartJson(scriptId);
  if (!fig) return;

  /* ================================================
    Decode binary data
   ================================================ */
  // This was the critical fix that got Plotly charts working in my Flask app

  function decodeBinaryData(obj) {
    // Base64 decode helper
    function base64ToArray(base64, dtype) {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }

      // Convert based on dtype
      if (dtype === 'i1') return new Int8Array(bytes.buffer);
      if (dtype === 'i2') return new Int16Array(bytes.buffer);
      if (dtype === 'i4') return new Int32Array(bytes.buffer);
      if (dtype === 'f4') return new Float32Array(bytes.buffer);
      if (dtype === 'f8') return new Float64Array(bytes.buffer);
      return bytes;
    }

    // Recursively process the object
    if (obj && typeof obj === 'object') {
      // Check if this is binary encoded data
      if (obj.bdata && obj.dtype) {
        const decoded = base64ToArray(obj.bdata, obj.dtype);
        return Array.from(decoded);
      }

      // Process arrays
      if (Array.isArray(obj)) {
        return obj.map(item => decodeBinaryData(item));
      }

      // Process objects
      const result = {};
      for (const key in obj) {
        result[key] = decodeBinaryData(obj[key]);
      }
      return result;
    }

    return obj;
  }

  // Decode the entire figure
  const decodedFig = decodeBinaryData(fig);

  console.groupCollapsed(`[DX] ${containerId} - Decoded Data`);
  console.log('Decoded figure:', decodedFig);

  // Show the actual values for verification
  if (decodedFig.data && decodedFig.data[0]) {
    console.log('X values:', decodedFig.data[0].x);
    console.log('Y values:', decodedFig.data[0].y);
    console.log('Text values:', decodedFig.data[0].text);
  }
  console.groupEnd();

  // NEW: Add autosize to layout
  decodedFig.layout.autosize = true;

  // NEW: Enhanced config for better responsiveness
  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(containerId, decodedFig.data, decodedFig.layout, config)
    .then(() => {
      console.info(`[DX] ✅ Rendered ${containerId}`);

      // Force resize without setting any inline styles
      setTimeout(function () {
        // NEW: Only log once for the first chart to avoid spam
        if (containerId === 'importance-chart') {
          console.log('[DX] === FULL WIDTH HIERARCHY ===');
          console.log('Window width:', window.innerWidth);
          
          // Trace up from the content div
          const contentDiv = document.querySelector('.content');
          if (contentDiv) {
            let current = contentDiv;
            let level = 0;
            
            while (current && level < 5) {
              const styles = window.getComputedStyle(current);
              console.log(`Level ${level} - ${current.tagName}.${current.className || '[no class]'}:`, {
                width: styles.width,
                maxWidth: styles.maxWidth,
                padding: styles.padding,
                margin: styles.margin,
                boxSizing: styles.boxSizing
              });
              current = current.parentElement;
              level++;
            }
          }
          
          // Check body styles specifically
          const bodyStyles = window.getComputedStyle(document.body);
          console.log('Body styles:', {
            width: bodyStyles.width,
            padding: bodyStyles.padding,
            margin: bodyStyles.margin
          });
        }
        
        Plotly.Plots.resize(containerId);
      }, 100);
    })
}


/* ======================================================================
   Initialize charts based on current page
   ======================================================================= */

document.addEventListener('DOMContentLoaded', () => {
  // Check which charts exist on the current page and render them

  // Survival Analysis Dashboard charts
  if (document.getElementById('class-chart')) {
    renderPlot('class-chart', 'class-chart-data');
    renderPlot('gender-chart', 'gender-chart-data');
    renderPlot('age-chart', 'age-chart-data');
    renderPlot('family-chart', 'family-chart-data');
  }

  // Model Performance Dashboard charts
  if (document.getElementById('importance-chart')) {
    renderPlot('importance-chart', 'importance-chart-data');
    renderPlot('confusion-chart', 'confusion-chart-data');
    renderPlot('roc-chart', 'roc-chart-data');
  }
});


/* ======================================================================
   Model Training Functionality
   ======================================================================= */

document.addEventListener('DOMContentLoaded', () => {
  // Check which charts exist on the current page and render them

  // Survival Analysis Dashboard charts
  if (document.getElementById('class-chart')) {
    renderPlot('class-chart', 'class-chart-data');
    renderPlot('gender-chart', 'gender-chart-data');
    renderPlot('age-chart', 'age-chart-data');
    renderPlot('family-chart', 'family-chart-data');
  }

  // Model Performance Dashboard charts
  if (document.getElementById('importance-chart')) {
    renderPlot('importance-chart', 'importance-chart-data');
    renderPlot('confusion-chart', 'confusion-chart-data');
    renderPlot('roc-chart', 'roc-chart-data');
  }

  // Training functionality (only on main dashboard)
  if (document.getElementById('train-model-btn')) {

    // Check model status on page load
    fetch('/model-info')
      .then(response => {
        const btnText = document.getElementById('btn-text');
        if (response.ok) {
          btnText.textContent = 'Retrain Model';
        } else {
          // Note: model-status element doesn't exist, but keeping for compatibility
          const modelStatus = document.getElementById('model-status');
          if (modelStatus) {
            modelStatus.textContent = '⚠ No model found';
            modelStatus.className = 'model-status-warning';
          }
          btnText.textContent = 'Train Initial Model';
        }
      });

    // Handle model training
    document.getElementById('train-model-btn').addEventListener('click', async function () {
      const btn = this;
      const btnText = document.getElementById('btn-text');
      const spinner = document.getElementById('btn-spinner');
      const message = document.getElementById('training-message');

      // Disable button and show spinner
      btn.disabled = true;
      btnText.textContent = 'Training...';
      spinner.classList.remove('hidden');
      message.textContent = 'Model in training 🏋️‍♂️';
      message.className = 'training-message training-in-progress';  // Keep both classes

      try {
        const response = await fetch('/retrain', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            tune_hyperparameters: true
          })
        });

        const result = await response.json();

        if (response.ok) {
          // Success
          message.innerHTML = `
            ✓ Model trained successfully!<br>
            Accuracy: ${result.metrics.accuracy}%<br>
            <a href="/model-performance" class="link">View Performance Metrics</a>
          `;
          message.className = 'training-message training-success';  // Keep both classes
          btnText.textContent = 'Retrain Model';
        } else {
          // Error
          message.textContent = `Error: ${result.error}`;
          message.className = 'training-message training-error';  // Keep both classes
          btnText.textContent = 'Try Again';
        }
      } catch (error) {
        message.textContent = `Error: ${error.message}`;
        message.className = 'training-message training-error';  // Keep both classes
        btnText.textContent = 'Try Again';
      } finally {
        // Re-enable button and hide spinner
        btn.disabled = false;
        spinner.classList.add('hidden');
      }
    });
  }
});


/* ======================================================================
    Tooltip Positioning
   ======================================================================= */

// NEW: Keep tooltips within viewport
document.addEventListener('mouseover', function (e) {
  if (e.target.classList.contains('info-icon')) {
    const tooltip = window.getComputedStyle(e.target, ':before');
    const rect = e.target.getBoundingClientRect();

    // Adjust tooltip position if it would go off screen
    if (rect.left < 160) {  // Too close to left edge
      e.target.style.setProperty('--tooltip-left', '0');
      e.target.style.setProperty('--tooltip-transform', 'translateY(-10px)');
    } else if (rect.right > window.innerWidth - 160) {  // Too close to right edge
      e.target.style.setProperty('--tooltip-left', 'auto');
      e.target.style.setProperty('--tooltip-right', '0');
      e.target.style.setProperty('--tooltip-transform', 'translateY(-10px)');
    }
  }
});


/* ======================================================================
    Set Charts to Fill Plot Area
   ======================================================================= */

// NEW: Make all charts responsive on window resize
window.addEventListener('resize', function () {
  // List all your chart container IDs
  const chartIds = [
    'class-chart',
    'gender-chart',
    'age-chart',
    'family-chart',
    'importance-chart',
    'confusion-chart',
    'roc-chart'
  ];

  chartIds.forEach(function (id) {
    if (document.getElementById(id)) {
      Plotly.Plots.resize(id);
    }
  });
});

// Force all charts to resize after everything is loaded (no inline styles)
window.addEventListener('load', function() {
  setTimeout(function() {
    const chartIds = [
      'class-chart', 
      'gender-chart', 
      'age-chart', 
      'family-chart',
      'importance-chart',
      'confusion-chart',
      'roc-chart'
    ];
    
    chartIds.forEach(function(id) {
      if (document.getElementById(id)) {
        Plotly.Plots.resize(id);
      }
    });
  }, 500);
});