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
   // This was the critical fix that 

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

  Plotly.newPlot(containerId, decodedFig.data, decodedFig.layout, { responsive: true })
        .then(() => console.info(`[DX] ✅ Rendered ${containerId}`))
        .catch(err  => console.error(`[DX] ❌ Plotly failed for ${containerId}:`, err));
}


document.addEventListener('DOMContentLoaded', () => {
  renderPlot('class-chart',   'class-chart-data');
  renderPlot('gender-chart',  'gender-chart-data');
  renderPlot('age-chart',     'age-chart-data');
  renderPlot('family-chart',  'family-chart-data');
});