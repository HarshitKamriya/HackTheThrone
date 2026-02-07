const API_BASE = CONFIG.API_BASE;

let currentSessionId = null;
const output = document.getElementById("output");

function log(data) {
  output.textContent = JSON.stringify(data, null, 2);
}

document.getElementById("startBtn").addEventListener("click", async () => {
  try {
    const response = await axios.post(`${API_BASE}/start`, {
      deviceInfo: navigator.userAgent
    });

    const data = response.data;
    currentSessionId = data.sessionId;
    log(data);
  } catch (error) {
    log({ error: error.message });
  }
});

document.getElementById("statusBtn").addEventListener("click", async () => {
  if (!currentSessionId) {
    return log({ error: "No active session" });
  }

  try {
    const response = await axios.get(
      `${API_BASE}/${currentSessionId}`
    );

    log(response.data);
  } catch (error) {
    log({ error: error.response?.data || error.message });
  }
});

document.getElementById("endBtn").addEventListener("click", async () => {
  if (!currentSessionId) {
    return log({ error: "No active session" });
  }

  try {
    const response = await axios.post(
      `${API_BASE}/end/${currentSessionId}`
    );

    log(response.data);
    currentSessionId = null;
  } catch (error) {
    log({ error: error.response?.data || error.message });
  }
});

