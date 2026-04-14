document.addEventListener('DOMContentLoaded', () => {
    const splash = document.getElementById('splash-screen');
    const appContainer = document.getElementById('app-container');
    const startBtn = document.getElementById('start-btn');
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const preview = document.getElementById('image-preview');
    const placeholder = document.getElementById('upload-placeholder');
    const analyzeBtn = document.getElementById('analyze-btn');
    const digitalText = document.getElementById('digital-text');
    const charMetrics = document.getElementById('char-metrics');
    const identityMetrics = document.getElementById('identity-metrics');
    const writerHeadline = document.getElementById('writer-headline');
    const neuralPatches = document.getElementById('neural-patches');
    const dictToggle = document.getElementById('dict-toggle');
    const rawWordHint = document.getElementById('raw-word-hint');

    // --- State ---
    let selectedFile = null;

    // --- Transitions ---
    startBtn.addEventListener('click', () => {
        splash.classList.add('fade-out');
        setTimeout(() => {
            splash.style.display = 'none';
            appContainer.classList.remove('hidden');
            appContainer.classList.add('fade-in');
        }, 800);
    });

    // --- File Handling ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent)';
        dropZone.style.background = '#f0f9ff';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--glass-border)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            placeholder.classList.add('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --- API Call ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Reset UI
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'ANALYZING...';
        digitalText.textContent = '...';
        digitalText.style.opacity = '0.5';
        charMetrics.innerHTML = '<p class="empty-hint">Processing...</p>';
        identityMetrics.innerHTML = '<p class="empty-hint">Processing...</p>';
        neuralPatches.innerHTML = '<p class="empty-hint">Processing...</p>';
        writerHeadline.textContent = '';
        writerHeadline.classList.add('hidden');
        rawWordHint.textContent = '';
        rawWordHint.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();

            // Comparison logic
            const useDict = dictToggle.checked;
            const finalWord = useDict ? data.word : data.raw_word;

            // Update UI
            digitalText.textContent = finalWord || '(No text detected)';
            digitalText.style.opacity = '1';

            // Show hint if autocorrect changed something (or if user wants to see correction)
            if (useDict && data.word.toLowerCase() !== data.raw_word.toLowerCase()) {
                rawWordHint.innerHTML = `Raw AI thought: <span>${data.raw_word}</span>`;
                rawWordHint.classList.remove('hidden');
            } else if (!useDict && data.word.toLowerCase() !== data.raw_word.toLowerCase()) {
                rawWordHint.innerHTML = `Correction available: <span>${data.word}</span>`;
                rawWordHint.classList.remove('hidden');
            }
            
            charMetrics.innerHTML = data.char_html || '<p>No character metrics.</p>';
            identityMetrics.innerHTML = data.writer_html || '<p>No identity metrics.</p>';
            
            if (data.writer_headline) {
                writerHeadline.textContent = data.writer_headline;
                writerHeadline.classList.remove('hidden');
            }

            // Render Neural Patches
            if (data.patches && data.patches.length > 0) {
                neuralPatches.innerHTML = '';
                data.patches.forEach((base64, idx) => {
                    const div = document.createElement('div');
                    div.className = 'character-patch animate-up';
                    div.style.animationDelay = `${idx * 0.05}s`;
                    
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${base64}`;
                    img.alt = `Patch ${idx}`;
                    
                    const label = document.createElement('div');
                    label.className = 'patch-label';
                    label.textContent = `P${idx}`;
                    
                    div.appendChild(img);
                    div.appendChild(label);
                    neuralPatches.appendChild(div);
                });
            } else {
                neuralPatches.innerHTML = '<p class="empty-hint">No patches extracted.</p>';
            }

        } catch (error) {
            console.error('Error:', error);
            digitalText.textContent = 'Error during analysis';
            charMetrics.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'RUN ANALYSIS';
        }
    });

    // Add extra CSS for the Python-generated writer fragments
    const style = document.createElement('style');
    style.innerHTML = `
        .writer-topk-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .writer-topk-name { font-weight: 600; font-size: 0.9rem; }
        .writer-topk-bar-bg { flex-grow: 1; height: 6px; background: #f1f5f9; border-radius: 3px; margin: 0 12px; }
        .writer-topk-bar-fill { height: 100%; background: var(--accent); border-radius: 3px; }
        .writer-topk-pct { font-weight: 700; font-size: 0.8rem; width: 40px; text-align: right; }
    `;
    document.head.appendChild(style);
});
