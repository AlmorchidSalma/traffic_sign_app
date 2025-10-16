// static/main.js
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const resultArea = document.getElementById("resultArea");
    const annotImg = document.getElementById("annotImg");
    const detectionsList = document.getElementById("detectionsList");
    const loading = document.getElementById("loading");
    const errorBox = document.getElementById("errorBox");
    const resetBtn = document.getElementById("resetBtn");
    const downloadBtn = document.getElementById("downloadBtn");
  
    resetBtn.onclick = () => {
      fileInput.value = "";
      resultArea.style.display = "none";
      errorBox.style.display = "none";
      detectionsList.innerHTML = "";
    };
  
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!fileInput.files.length) return;
      const file = fileInput.files[0];
      const fd = new FormData();
      fd.append("file", file);
  
      loading.style.display = "inline-block";
      errorBox.style.display = "none";
      resultArea.style.display = "none";
  
      try {
        const resp = await fetch("/predict", { method: "POST", body: fd });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.error || resp.statusText);
        }
        const data = await resp.json();
  
        if (data.image) {
          annotImg.src = data.image;
          // set download
          downloadBtn.href = data.image;
          downloadBtn.style.display = "inline-block";
        } else {
          annotImg.src = "";
          downloadBtn.style.display = "none";
        }
  
        detectionsList.innerHTML = "";
        if (data.detections && data.detections.length) {
          data.detections.forEach(d => {
            const li = document.createElement("div");
            li.className = "list-group-item";
            li.innerHTML = `<strong>${d.label}</strong> — conf: ${d.confidence.toFixed(2)} — box: [${d.box.map(x=>Math.round(x)).join(", ")}]`;
            detectionsList.appendChild(li);
          });
        } else if (data.detections_summary) {
          const li = document.createElement("div");
          li.className = "list-group-item";
          li.innerText = `Video processed. ${data.detections_summary.length} detections (see server response).`;
          detectionsList.appendChild(li);
          // For videos, server returns base64 video (large) — you could create <video> element to play it:
          if (data.video_base64) {
            const v = document.createElement("video");
            v.controls = true;
            v.src = "data:video/mp4;base64," + data.video_base64;
            v.className = "w-100 mt-3";
            detectionsList.appendChild(v);
          }
        } else {
          const li = document.createElement("div");
          li.className = "list-group-item";
          li.innerText = "No detections.";
          detectionsList.appendChild(li);
        }
  
        resultArea.style.display = "block";
      } catch (err) {
        console.error(err);
        errorBox.innerText = err.message || "Unknown error";
        errorBox.style.display = "block";
      } finally {
        loading.style.display = "none";
      }
    });
  });
  