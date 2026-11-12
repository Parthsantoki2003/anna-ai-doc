        const API_BASE = 'https://anna-document-workspace-wj7amu7tcq-uc.a.run.app';
        let currentConversationId = null;
        let conversationHistory = [];
        let documentsLoaded = [];
        let uploadInProgress = false;
        let selectedFile = null;
        let anna = null;

        function getUserId() {
            let uid = null;
            try {
                uid = localStorage.getItem('anna_doc_user_id');
            } catch (e) {}
            if (!uid) {
                if (anna && anna.user && anna.user.id) {
                    uid = String(anna.user.id);
                } else if (anna && anna.account && anna.account.id) {
                    uid = String(anna.account.id);
                } else {
                    const rnd = (typeof crypto !== 'undefined' && crypto.randomUUID)
                        ? crypto.randomUUID().replace(/-/g, '')
                        : Math.random().toString(36).substring(2) + Date.now().toString(36);
                    uid = 'usr_' + rnd;
                }
                try {
                    localStorage.setItem('anna_doc_user_id', uid);
                } catch (e) {}
            }
            return uid;
        }

        async function initAnnaRuntime() {
            try {
                let Runtime = typeof window !== 'undefined' ? window.AnnaAppRuntime : null;
                if (!Runtime) {
                    const sdk = await import('/static/anna-apps/_sdk/latest/index.js');
                    Runtime = sdk.AnnaAppRuntime || sdk.default;
                }
                if (Runtime && typeof Runtime.connect === 'function') {
                    anna = await Runtime.connect();
                    await anna.window.set_title({ title: 'DocuMind' }).catch(() => {});
                }
            } catch (e) {
                console.warn('Anna Runtime not available (standalone mode):', e);
            }
        }

        function setupEventListeners() {
            document.getElementById('fileInput').addEventListener('change', handleFileSelect);
            document.getElementById('uploadActionBtn').addEventListener('click', uploadSelectedFile);
            document.getElementById('newConversationBtn').addEventListener('click', startNewConversation);
            document.getElementById('queryForm').addEventListener('submit', handleQuerySubmit);
            
            const dropzone = document.querySelector('[data-dropzone]');
            if (dropzone) {
                dropzone.addEventListener('dragover', handleDragOver);
                dropzone.addEventListener('dragleave', handleDragLeave);
                dropzone.addEventListener('drop', handleDrop);
                dropzone.addEventListener('click', (e) => {
                    if (e.target.id !== 'fileInput') {
                        document.getElementById('fileInput')?.click();
                    }
                });
            }

            // Delegated click handler for deleting documents
            const docList = document.getElementById('documentsList');
            if (docList) {
                docList.addEventListener('click', (e) => {
                    const btn = e.target.closest('[data-delete]');
                    if (!btn) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const fid = btn.getAttribute('data-delete');
                    const fname = btn.getAttribute('data-name') || 'Document';
                    openDeleteModal(fid, fname);
                });
            }

            // Delete modal controls
            document.getElementById('cancelDeleteBtn')?.addEventListener('click', closeDeleteModal);
            document.getElementById('confirmDeleteBtn')?.addEventListener('click', executeDelete);
            document.getElementById('deleteModal')?.addEventListener('click', (e) => {
                if (e.target.id === 'deleteModal') closeDeleteModal();
            });

            // Workflow Tab Switcher
            initWorkflowTabs();

            // Workflow trigger buttons
            document.getElementById('runSummaryBtn')?.addEventListener('click', runExecutiveSummary);
            document.getElementById('runRiskAuditBtn')?.addEventListener('click', runRiskAudit);
            document.getElementById('runExtractBtn')?.addEventListener('click', runDataExtract);
            document.getElementById('runCompareBtn')?.addEventListener('click', runCompareDocuments);
            document.getElementById('runQuizBtn')?.addEventListener('click', runQuiz);

            // Evidence modal controls
            document.getElementById('closeEvidenceBtn')?.addEventListener('click', closeEvidenceModal);
            document.getElementById('dismissEvidenceBtn')?.addEventListener('click', closeEvidenceModal);
            document.getElementById('copyEvidenceBtn')?.addEventListener('click', copyEvidenceText);
            document.getElementById('evidenceModal')?.addEventListener('click', (e) => {
                if (e.target.id === 'evidenceModal') closeEvidenceModal();
            });

            // Export report modal controls
            document.getElementById('exportReportBtn')?.addEventListener('click', openExportModal);
            document.getElementById('closeExportBtn')?.addEventListener('click', closeExportModal);
            document.getElementById('copyReportBtn')?.addEventListener('click', copyReportText);
            document.getElementById('copyDataCsvBtn')?.addEventListener('click', copyExtractedCsv);
            document.getElementById('exportModal')?.addEventListener('click', (e) => {
                if (e.target.id === 'exportModal') closeExportModal();
            });
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    showNotification('Only PDF files are supported', 'error');
                    return;
                }
                selectedFile = file;
                document.getElementById('fileNameDisplay').textContent = `Selected: ${file.name}`;
                showNotification(`Selected ${file.name}. Click "Upload PDF" to proceed.`, 'info');
            }
        }

        function uploadSelectedFile(e) {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            const input = document.getElementById('fileInput');
            const file = selectedFile || (input.files && input.files[0]);
            if (file) {
                uploadFile(file);
            } else {
                showNotification('Please select a PDF document first', 'warning');
                input.click();
            }
        }

        function handleDragOver(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('bg-indigo-50', 'border-indigo-400');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('bg-indigo-50', 'border-indigo-400');
        }

        function handleDrop(event) {
            event.preventDefault();
            event.stopPropagation();
            this.classList.remove('bg-indigo-50', 'border-indigo-400');
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    showNotification('Only PDF files are supported', 'error');
                    return;
                }
                selectedFile = file;
                document.getElementById('fileNameDisplay').textContent = `Selected: ${file.name}`;
                showNotification(`Selected ${file.name}. Click "Upload PDF" to proceed.`, 'info');
            }
        }

        async function annaHostUpload(file) {
            if (!anna) return null;
            try {
                if (file.size <= 8 * 1024 * 1024) {
                    const bytes = new Uint8Array(await file.arrayBuffer());
                    let binary = '';
                    const step = 0x8000;
                    for (let i = 0; i < bytes.length; i += step) {
                        binary += String.fromCharCode(...bytes.subarray(i, i + step));
                    }
                    const b64 = btoa(binary);
                    const response = await anna.upload.inline({
                        filename: file.name,
                        mime_type: 'application/pdf',
                        content_b64: b64,
                        bytes_b64: b64,
                        mime: 'application/pdf',
                        purpose: 'document-ingestion'
                    });
                    const payload = response?.result ?? response;
                    return payload?.url || payload?.get_url || payload?.download_url || null;
                }

                const negotiated = await anna.upload.negotiate({
                    filename: file.name,
                    mime_type: 'application/pdf',
                    mime: 'application/pdf',
                    purpose: 'document-ingestion',
                    size_bytes: file.size
                });
                const negPayload = negotiated?.result ?? negotiated;
                const uploadUrl = negPayload?.presigned_url || negPayload?.upload_url;
                const r2Key = negPayload?.r2_key || negPayload?.file_ref;
                if (!uploadUrl || !r2Key) return null;

                const uploadResponse = await fetch(uploadUrl, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/pdf' },
                    body: file
                });
                if (!uploadResponse.ok) return null;

                const confirmed = await anna.upload.confirm({
                    r2_key: r2Key,
                    file_ref: r2Key
                });
                const confPayload = confirmed?.result ?? confirmed;
                return confPayload?.url || confPayload?.get_url || confPayload?.download_url || null;
            } catch (error) {
                console.warn('Anna host upload fallback to direct ingest:', error);
                return null;
            }
        }

        async function uploadFile(file) {
            if (uploadInProgress) return;
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showNotification('Only PDF files are supported', 'error');
                return;
            }
            uploadInProgress = true;
            const uploadBtn = document.getElementById('uploadActionBtn');
            if (uploadBtn) uploadBtn.disabled = true;

            document.getElementById('fileProgress').classList.remove('hidden');
            document.getElementById('fileNameDisplay').textContent = `Uploading ${file.name}...`;

            try {
                const hostedUrl = await annaHostUpload(file);
                let response;

                if (hostedUrl && /^https?:\/\//i.test(hostedUrl)) {
                    response = await fetch(`${API_BASE}/ingest-from-url`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-User-Id': getUserId()
                        },
                        body: JSON.stringify({
                            file_url: hostedUrl,
                            file_name: file.name,
                            chunk_strategy: 'recursive',
                            chunk_size: 500,
                            chunk_overlap: 50
                        })
                    });
                } else {
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('chunk_strategy', 'recursive');
                    formData.append('chunk_size', 500);
                    formData.append('chunk_overlap', 50);

                    response = await fetch(`${API_BASE}/ingest`, {
                        method: 'POST',
                        headers: {
                            'X-User-Id': getUserId()
                        },
                        body: formData
                    });
                }

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Upload failed');
                }

                const data = await response.json();
                showNotification(`Document uploaded: ${data.file_name} (${data.chunks_created} chunks)`, 'success');
                documentsLoaded = [
                    ...documentsLoaded.filter(doc => doc.file_id !== data.file_id),
                    {
                        file_id: data.file_id,
                        file_name: data.file_name,
                        chunks_count: data.chunks_created,
                        upload_time: new Date().toISOString()
                    }
                ];
                renderDocuments(documentsLoaded);
                
                selectedFile = null;
                document.getElementById('fileNameDisplay').textContent = 'Choose a PDF or drag it here';
                document.getElementById('fileInput').value = '';
                document.getElementById('fileProgress').classList.add('hidden');
                
                loadStats();
                enableQueryInput();
            } catch (error) {
                showNotification(`Upload failed: ${error.message}`, 'error');
                document.getElementById('fileProgress').classList.add('hidden');
            } finally {
                uploadInProgress = false;
                if (uploadBtn) uploadBtn.disabled = false;
            }
        }

        async function loadDocuments() {
            try {
                const response = await fetch(`${API_BASE}/documents`, {
                    headers: {
                        'X-User-Id': getUserId()
                    }
                });
                if (!response.ok) throw new Error(`Documents request failed (${response.status})`);
                const documents = await response.json();
                documentsLoaded = documents;
                renderDocuments(documents);
            } catch (error) {
                console.error('Failed to load documents:', error);
            }
        }

        let pendingDeleteId = null;
        let pendingDeleteName = null;

        function openDeleteModal(fileId, fileName) {
            pendingDeleteId = fileId;
            pendingDeleteName = fileName;
            const modal = document.getElementById('deleteModal');
            const textEl = document.getElementById('deleteModalText');
            if (textEl) {
                textEl.textContent = `Are you sure you want to delete "${fileName}"? This will remove it from your workspace and clear its search index.`;
            }
            if (modal) modal.classList.remove('hidden');
        }

        function closeDeleteModal() {
            pendingDeleteId = null;
            pendingDeleteName = null;
            const modal = document.getElementById('deleteModal');
            if (modal) modal.classList.add('hidden');
            const confirmBtn = document.getElementById('confirmDeleteBtn');
            if (confirmBtn) {
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = '<i class="fa-solid fa-trash-alt pointer-events-none"></i><span>Delete</span>';
            }
        }

        async function executeDelete() {
            if (!pendingDeleteId) return;
            const fileId = pendingDeleteId;
            const fileName = pendingDeleteName || 'Document';
            const confirmBtn = document.getElementById('confirmDeleteBtn');
            if (confirmBtn) {
                confirmBtn.disabled = true;
                confirmBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin pointer-events-none"></i><span>Deleting...</span>';
            }

            try {
                const response = await fetch(`${API_BASE}/documents/${fileId}`, {
                    method: 'DELETE',
                    headers: {
                        'X-User-Id': getUserId()
                    }
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || 'Delete failed');
                }

                showNotification(`Document "${fileName}" deleted`, 'success');
                documentsLoaded = documentsLoaded.filter(doc => doc.file_id !== fileId);
                renderDocuments(documentsLoaded);
                enableQueryInput();
                loadStats();
                await loadDocuments();
            } catch (error) {
                console.error('Delete failed:', error);
                showNotification(`Delete failed: ${error.message}`, 'error');
            } finally {
                closeDeleteModal();
            }
        }

        function renderDocuments(documents) {
            const list = document.getElementById('documentsList');
            if (!list) return;

            const badge = document.getElementById('docCountBadge');
            if (badge) badge.textContent = documents.length;

            // Populate compare selectors
            updateCompareDropdowns(documents);

            // Enable or disable chat input based on documents
            const hasDocs = documents.length > 0;
            const queryInput = document.getElementById('queryInput');
            const submitBtn = document.getElementById('submitBtn');
            const chatStatus = document.getElementById('chatStatus');

            if (queryInput) queryInput.disabled = !hasDocs;
            if (submitBtn) submitBtn.disabled = !hasDocs;
            if (chatStatus) {
                chatStatus.textContent = hasDocs 
                    ? `${documents.length} document${documents.length > 1 ? 's' : ''} indexed • Ready for reasoning`
                    : 'Upload a PDF to unlock deep reasoning workflows';
            }

            if (documents.length === 0) {
                list.innerHTML = `
                    <div class="text-center py-8 text-slate-400">
                        <i class="fa-regular fa-file-pdf text-3xl mb-2 text-slate-300"></i>
                        <p class="text-xs">No documents uploaded yet</p>
                    </div>
                `;
                return;
            }

            list.innerHTML = documents.map(doc => `
                <div class="doc-item group">
                    <div class="flex-1 min-w-0 pr-1">
                        <div class="flex items-center gap-1.5 mb-0.5">
                            <span class="inline-flex items-center px-1.5 py-0.2 rounded text-[9px] font-extrabold bg-rose-50 text-rose-600 border border-rose-200">PDF</span>
                            <p class="text-xs font-semibold text-slate-800 truncate" title="${escapeHtml(doc.file_name)}">
                                ${escapeHtml(doc.file_name)}
                            </p>
                        </div>
                        <p class="text-[10px] text-slate-400 font-medium pl-0.5">${doc.chunks_count} chunks indexed</p>
                    </div>
                    <button type="button" class="text-slate-400 hover:text-rose-600 hover:bg-rose-50 w-7 h-7 rounded-lg flex items-center justify-center transition cursor-pointer flex-shrink-0" title="Delete document" data-delete="${escapeHtml(doc.file_id)}" data-name="${escapeHtml(doc.file_name)}">
                        <i class="fa-regular fa-trash-can text-xs pointer-events-none"></i>
                    </button>
                </div>
            `).join('');
        }

        async function loadStats() {
            try {
                const response = await fetch(`${API_BASE}/stats`, {
                    headers: {
                        'X-User-Id': getUserId()
                    }
                });
                if (!response.ok) return;
                const stats = await response.json();
                const chunksEl = document.getElementById('statsChunks');
                if (chunksEl) chunksEl.textContent = stats.total_chunks ?? 0;
                const docsEl = document.getElementById('statsDocuments');
                if (docsEl) docsEl.textContent = stats.total_documents ?? 0;
                const badge = document.getElementById('docCountBadge');
                if (badge && stats.total_documents !== undefined) badge.textContent = stats.total_documents;
            } catch (error) {
                console.warn('Stats fetch skipped:', error);
            }
        }

        function startNewConversation() {
            currentConversationId = null;
            conversationHistory = [];
            document.getElementById('messagesContainer').innerHTML = `
                <div class="text-center py-12">
                    <i class="fa-solid fa-sparkles text-6xl text-slate-300 mb-4"></i>
                    <p class="text-slate-500 text-lg">Upload a document and ask questions</p>
                </div>
            `;
            const convHistory = document.getElementById('conversationHistory');
            if (convHistory) {
                convHistory.innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No conversation yet</p>';
            }
            enableQueryInput();
        }

        async function handleQuerySubmit(event) {
            event.preventDefault();
            
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            if (documentsLoaded.length === 0) {
                showNotification('Please upload a document first', 'warning');
                return;
            }

            addMessageToUI('user', query);
            document.getElementById('queryInput').value = '';
            document.getElementById('submitBtn').disabled = true;

            const loadingId = addLoadingMessage();

            try {
                const response = await fetch(`${API_BASE}/query`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-User-Id': getUserId()
                    },
                    body: JSON.stringify({
                        query,
                        top_k: 5,
                        generate_answer: true,
                        conversation_context: conversationHistory.slice(-6)
                    })
                });

                if (!response.ok) throw new Error('Query failed');

                const data = await response.json();
                
                document.getElementById(loadingId)?.remove();

                let answer = data.answer;
                if (data.fallback_required) {
                    if (anna && anna.llm && typeof anna.llm.complete === 'function') {
                        try {
                            showNotification('Groq limit reached: generating answer via Anna LLM failover...', 'info');
                            const fallbackRes = await anna.llm.complete({
                                systemPrompt: data.fallback_system_prompt,
                                messages: [
                                    { role: 'user', content: data.fallback_user_prompt }
                                ]
                            });
                            answer = fallbackRes?.text ||
                                     (typeof fallbackRes?.content === 'string' ? fallbackRes.content : null) ||
                                     (Array.isArray(fallbackRes?.content) && fallbackRes.content[0]?.text) ||
                                     fallbackRes?.result ||
                                     fallbackRes?.message?.content ||
                                     (typeof fallbackRes === 'string' ? fallbackRes : null);

                            if (!answer) throw new Error("Empty response from Anna LLM");
                            showNotification("Answer generated via Anna LLM failover", "success");
                        } catch (fallbackErr) {
                            console.warn("Anna LLM fallback failed:", fallbackErr);
                            answer = "I retrieved the relevant source documents below, but could not complete the AI summary due to temporary provider limits. Please review the cited excerpts.";
                        }
                    } else {
                        answer = "Primary LLM limit reached and Anna platform LLM is not available in standalone mode. Please review the relevant sources below.";
                    }
                }

                conversationHistory.push({
                    role: 'user',
                    query,
                    timestamp: new Date().toISOString()
                });

                conversationHistory.push({
                    role: 'assistant',
                    answer: answer,
                    sources: data.sources,
                    timestamp: new Date().toISOString()
                });

                currentConversationId = data.conversation_id;

                addAssistantMessage(answer, data.sources, data.processing_time);
                updateConversationHistory();

            } catch (error) {
                document.getElementById(loadingId)?.remove();
                showNotification(`Query failed: ${error.message}`, 'error');
            } finally {
                document.getElementById('submitBtn').disabled = false;
            }
        }

        function addMessageToUI(role, content) {
            const container = document.getElementById('messagesContainer');
            
            if (container.querySelector('.text-center')) {
                container.innerHTML = '';
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message-enter flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
            messageDiv.innerHTML = `
                <div class="chat-bubble ${role} ${role === 'user' ? 'text-white' : 'text-slate-900'}">
                    <div class="chat-meta"><span class="avatar">${role === 'user' ? 'Y' : 'A'}</span>${role === 'user' ? 'You' : 'Anna'}</div>
                    <p class="chat-content">${escapeHtml(content)}</p>
                </div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        function addLoadingMessage() {
            const container = document.getElementById('messagesContainer');
            const id = `loading-${Date.now()}`;
            
            const messageDiv = document.createElement('div');
            messageDiv.id = id;
            messageDiv.className = 'message-enter flex justify-start';
            messageDiv.innerHTML = `
                <div class="chat-bubble assistant">
                    <div class="typing-bubble">
                        <div class="w-2 h-2 bg-slate-400 rounded-full pulse-dot"></div>
                        <div class="w-2 h-2 bg-slate-400 rounded-full pulse-dot"></div>
                        <div class="w-2 h-2 bg-slate-400 rounded-full pulse-dot"></div>
                        <span class="typing-label">Anna is thinking...</span>
                    </div>
                </div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
            
            return id;
        }

        function addAssistantMessage(answer, sources, processingTime, groundingScore = 96) {
            const container = document.getElementById('messagesContainer');
            
            // Update top header grounding badge
            const gBadge = document.getElementById('groundingBadge');
            const gText = document.getElementById('groundingScoreText');
            if (gBadge && gText) {
                gBadge.classList.remove('hidden');
                gBadge.classList.add('flex');
                gText.textContent = `${groundingScore}% Grounded`;
            }

            const sourcesList = Array.isArray(sources) ? sources : [];
            let sourcesHtml = '';
            if (sourcesList.length > 0) {
                sourcesHtml = `
                    <div class="mt-3 pt-3 border-t border-slate-100">
                        <p class="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-1.5 flex items-center gap-1">
                            <i class="fa-solid fa-quote-left text-indigo-500"></i> Verified Citations (click to inspect):
                        </p>
                        <div class="flex flex-wrap gap-1.5">
                            ${sourcesList.map((src, i) => `
                                <button type="button" class="evidence-chip px-2.5 py-1 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 text-indigo-700 rounded-lg text-xs font-medium transition flex items-center gap-1.5 shadow-2xs" data-src-file="${escapeHtml(src.source || 'Document')}" data-src-text="${escapeHtml(src.text || '')}">
                                    <i class="fa-solid fa-file-pdf text-red-500 text-[10px]"></i>
                                    <span class="truncate max-w-[200px]">${escapeHtml(src.source || 'Source')}</span>
                                </button>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message-enter flex justify-start';
            
            messageDiv.innerHTML = `
                <div class="chat-bubble assistant max-w-2xl">
                    <div class="chat-meta flex items-center justify-between">
                        <div class="flex items-center gap-1.5">
                            <span class="avatar">D</span>DocuMind
                        </div>
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200">
                            ✦ ${groundingScore}% Grounded
                        </span>
                    </div>
                    <div class="chat-content leading-relaxed mb-2 text-slate-800">${renderMarkdown(answer || 'No answer generated')}</div>
                    ${sourcesHtml}
                    <p class="text-[10px] text-slate-400 mt-2.5">Synthesized in ${processingTime.toFixed(2)}s via hybrid retrieval</p>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;

            // Attach evidence chip click handlers
            messageDiv.querySelectorAll('.evidence-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    openEvidenceModal(chip.dataset.srcFile, chip.dataset.srcText);
                });
            });
        }

        function updateConversationHistory() {
            const container = document.getElementById('conversationHistory');
            if (!container) return;
            
            if (conversationHistory.length === 0) {
                container.innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No conversation yet</p>';
                return;
            }

            container.innerHTML = conversationHistory
                .filter(msg => msg.role === 'user')
                .slice(-5)
                .map(msg => `
                    <div class="p-3 bg-indigo-50 rounded-lg hover:bg-indigo-100 cursor-pointer transition" data-query="${escapeHtml(msg.query)}">
                        <p class="text-xs font-semibold text-indigo-900 truncate">${escapeHtml(msg.query)}</p>
                    </div>
                `).join('');

            container.querySelectorAll('[data-query]').forEach(el => {
                el.addEventListener('click', () => {
                    document.getElementById('queryInput').value = el.dataset.query;
                    document.getElementById('queryInput').focus();
                });
            });
        }

        function enableQueryInput() {
            const hasDocuments = documentsLoaded.length > 0;
            document.getElementById('queryInput').disabled = !hasDocuments;
            document.getElementById('submitBtn').disabled = !hasDocuments;
            
            if (hasDocuments) {
                document.getElementById('chatStatus').textContent = `${documentsLoaded.length} document(s) ready • Ask away!`;
            }
        }

        function showNotification(message, type = 'info') {
            const container = document.getElementById('notificationContainer');
            
            const colors = {
                success: 'bg-green-500',
                error: 'bg-red-500',
                warning: 'bg-yellow-500',
                info: 'bg-blue-500'
            };

            const notification = document.createElement('div');
            notification.className = `toast-notification ${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-3 pointer-events-auto`;
            notification.innerHTML = `
                <i class="fa-solid fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation' : type === 'warning' ? 'triangle-exclamation' : 'info'}-circle"></i>
                <span>${escapeHtml(message)}</span>
            `;

            container.appendChild(notification);
            setTimeout(() => notification.remove(), 4000);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function renderMarkdown(text) {
            const safe = escapeHtml(text);
            return safe
                .replace(/^### (.*)$/gm, '<h3>$1</h3>')
                .replace(/^## (.*)$/gm, '<h2>$1</h2>')
                .replace(/^# (.*)$/gm, '<h1>$1</h1>')
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
                .replace(/`([^`\n]+)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }


        // ========================================================
        // AI-Native Workflows Studio Implementation
        // ========================================================

        let currentActiveTab = 'chat';
        let latestWorkflowData = null;
        let lastReportMarkdown = '';
        let currentExtractedPoints = [];
        let activeQuizQuestions = [];
        let currentQuizScore = 0;
        let answeredQuestions = new Set();

        function initWorkflowTabs() {
            const tabs = document.querySelectorAll('#workflowTabs .workflow-tab-btn, #workflowTabs .tab-btn');
            tabs.forEach(btn => {
                btn.addEventListener('click', () => {
                    const target = btn.dataset.tab;
                    switchWorkflowTab(target);
                });
            });
        }

        function switchWorkflowTab(tabKey) {
            currentActiveTab = tabKey;
            
            // Update Tab button active styles
            document.querySelectorAll('#workflowTabs .workflow-tab-btn, #workflowTabs .tab-btn').forEach(b => {
                const isCurrent = b.dataset.tab === tabKey;
                if (isCurrent) {
                    b.className = `workflow-tab-btn active-${tabKey}`;
                } else {
                    b.className = 'workflow-tab-btn inactive';
                }
            });

            // Toggle views
            const viewMap = {
                chat: 'chatView',
                summary: 'summaryView',
                risks: 'risksView',
                extract: 'extractView',
                compare: 'compareView',
                quiz: 'quizView'
            };

            Object.entries(viewMap).forEach(([k, viewId]) => {
                const el = document.getElementById(viewId);
                if (el) {
                    if (k === tabKey) {
                        el.classList.remove('hidden');
                        if (k === 'chat') el.classList.add('flex');
                    } else {
                        el.classList.add('hidden');
                        el.classList.remove('flex');
                    }
                }
            });
        }

        function updateCompareDropdowns(documents) {
            const sel1 = document.getElementById('compareDoc1');
            const sel2 = document.getElementById('compareDoc2');
            if (!sel1 || !sel2) return;

            if (documents.length === 0) {
                sel1.innerHTML = '<option value="">No documents uploaded</option>';
                sel2.innerHTML = '<option value="">No documents uploaded</option>';
                return;
            }

            const options1 = documents.map(d => `<option value="${escapeHtml(d.file_id)}">${escapeHtml(d.file_name)}</option>`).join('');
            sel1.innerHTML = options1;

            const options2 = documents.map((d, i) => `<option value="${escapeHtml(d.file_id)}" ${i === 1 ? 'selected' : ''}>${escapeHtml(d.file_name)}</option>`).join('');
            sel2.innerHTML = options2;
        }

        // --- Workflow 1: Executive Briefing ---
        async function runExecutiveSummary() {
            if (documentsLoaded.length === 0) {
                showNotification('Please upload a PDF document first', 'warning');
                return;
            }

            const btn = document.getElementById('runSummaryBtn');
            const container = document.getElementById('summaryContent');
            if (btn) btn.disabled = true;

            container.innerHTML = `
                <div class="p-8 text-center space-y-3">
                    <i class="fa-solid fa-circle-notch fa-spin text-4xl text-indigo-600"></i>
                    <p class="text-sm font-semibold text-slate-700">Synthesizing Executive Briefing across document chunks...</p>
                    <p class="text-xs text-slate-500">Decomposing core thesis, findings, decisions, and action items.</p>
                </div>
            `;

            try {
                const resp = await fetch(`${API_BASE}/workflow/executive-summary`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-User-Id': getUserId() },
                    body: JSON.stringify({})
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || 'Summary generation failed');
                }

                const data = await resp.json();
                latestWorkflowData = { type: 'summary', data };
                lastReportMarkdown = compileSummaryReport(data);

                container.innerHTML = `
                    <div class="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-2xl border border-indigo-100 space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs uppercase font-bold text-indigo-600 tracking-wider">Executive Overview</span>
                            <span class="text-xs text-slate-500">Processed in ${data.processing_time.toFixed(2)}s</span>
                        </div>
                        <h4 class="text-xl font-bold text-slate-900">${escapeHtml(data.title)}</h4>
                        <p class="text-sm text-slate-700 leading-relaxed">${renderMarkdown(data.executive_summary)}</p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
                                <i class="fa-solid fa-lightbulb text-amber-500"></i> Key Findings
                            </h5>
                            <ul class="space-y-2 text-xs text-slate-600">
                                ${(data.key_findings || []).map(f => `<li class="flex items-start gap-2"><span class="text-indigo-500 font-bold">•</span><span>${escapeHtml(f)}</span></li>`).join('')}
                            </ul>
                        </div>

                        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
                                <i class="fa-solid fa-gavel text-indigo-600"></i> Critical Decisions
                            </h5>
                            <ul class="space-y-2 text-xs text-slate-600">
                                ${(data.critical_decisions || []).map(d => `<li class="flex items-start gap-2"><span class="text-purple-500 font-bold">•</span><span>${escapeHtml(d)}</span></li>`).join('')}
                            </ul>
                        </div>

                        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
                                <i class="fa-solid fa-clock-rotate-left text-blue-500"></i> Timeline & Milestones
                            </h5>
                            <ul class="space-y-2 text-xs text-slate-600">
                                ${(data.timeline_or_milestones || []).map(m => `<li class="flex items-start gap-2"><span class="text-blue-500 font-bold">•</span><span>${escapeHtml(m)}</span></li>`).join('')}
                            </ul>
                        </div>
                    </div>

                    ${(data.action_items && data.action_items.length > 0) ? `
                        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
                                <i class="fa-solid fa-list-check text-emerald-600"></i> Immediate Action Items
                            </h5>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-700">
                                ${data.action_items.map(a => `
                                    <div class="flex items-center gap-2.5 p-2 bg-slate-50 rounded-lg border border-slate-100">
                                        <i class="fa-regular fa-square-check text-indigo-600 text-sm"></i>
                                        <span>${escapeHtml(a)}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                `;
                showNotification('Executive Briefing generated successfully', 'success');
            } catch (err) {
                console.error(err);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-sm">Briefing generation failed: ${escapeHtml(err.message)}</div>`;
                showNotification(`Failed: ${err.message}`, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        // --- Workflow 2: Risk Audit ---
        async function runRiskAudit() {
            if (documentsLoaded.length === 0) {
                showNotification('Please upload a PDF document first', 'warning');
                return;
            }

            const btn = document.getElementById('runRiskAuditBtn');
            const container = document.getElementById('riskAuditContent');
            if (btn) btn.disabled = true;

            container.innerHTML = `
                <div class="p-8 text-center space-y-3">
                    <i class="fa-solid fa-circle-notch fa-spin text-4xl text-red-600"></i>
                    <p class="text-sm font-semibold text-slate-700">Scanning document for contractual, financial, and compliance risks...</p>
                    <p class="text-xs text-slate-500">Checking liability caps, indemnities, penalties, and missing standard clauses.</p>
                </div>
            `;

            try {
                const resp = await fetch(`${API_BASE}/workflow/risk-audit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-User-Id': getUserId() },
                    body: JSON.stringify({})
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || 'Risk audit failed');
                }

                const data = await resp.json();
                latestWorkflowData = { type: 'risks', data };
                lastReportMarkdown = compileRiskReport(data);

                const severityColors = {
                    High: 'bg-red-50 text-red-700 border-red-200',
                    Medium: 'bg-amber-50 text-amber-700 border-amber-200',
                    Low: 'bg-emerald-50 text-emerald-700 border-emerald-200'
                };

                container.innerHTML = `
                    <div class="p-5 rounded-2xl bg-white border border-slate-200 shadow-2xs flex items-center justify-between">
                        <div class="flex items-center gap-4">
                            <div class="w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold ${data.overall_risk_level === 'High' ? 'bg-red-100 text-red-700' : data.overall_risk_level === 'Medium' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}">
                                <i class="fa-solid fa-triangle-exclamation"></i>
                            </div>
                            <div>
                                <div class="flex items-center gap-2">
                                    <h4 class="text-base font-bold text-slate-900">Overall Exposure: ${escapeHtml(data.overall_risk_level)}</h4>
                                    <span class="text-xs px-2.5 py-0.5 rounded-full border ${severityColors[data.overall_risk_level] || 'bg-slate-100 text-slate-700'} font-bold">
                                        ${data.risks.length} Risk Points Found
                                    </span>
                                </div>
                                <p class="text-xs text-slate-600 mt-1">${escapeHtml(data.risk_score_narrative)}</p>
                            </div>
                        </div>
                    </div>

                    <div class="space-y-3">
                        <h5 class="text-xs font-bold uppercase tracking-wider text-slate-600">Risk Severity Matrix</h5>
                        <div class="grid grid-cols-1 gap-3">
                            ${data.risks.map(r => `
                                <div class="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-2 hover:border-slate-300 transition">
                                    <div class="flex items-center justify-between">
                                        <div class="flex items-center gap-2">
                                            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${severityColors[r.severity] || 'bg-slate-100 text-slate-700'}">
                                                ${escapeHtml(r.severity)} Severity
                                            </span>
                                            <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                                                ${escapeHtml(r.category)}
                                            </span>
                                            <h6 class="text-sm font-bold text-slate-900">${escapeHtml(r.title)}</h6>
                                        </div>
                                    </div>
                                    <div class="p-2.5 bg-slate-50 rounded-lg text-xs text-slate-700 font-mono border border-slate-100">
                                        <span class="text-slate-400 select-none">"</span>${escapeHtml(r.clause_excerpt)}<span class="text-slate-400 select-none">"</span>
                                    </div>
                                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 text-xs">
                                        <div>
                                            <span class="font-bold text-slate-700">Business Impact:</span>
                                            <p class="text-slate-600 mt-0.5">${escapeHtml(r.impact)}</p>
                                        </div>
                                        <div>
                                            <span class="font-bold text-emerald-700">Recommended Mitigation:</span>
                                            <p class="text-slate-600 mt-0.5">${escapeHtml(r.mitigation)}</p>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    ${(data.missing_clauses && data.missing_clauses.length > 0) ? `
                        <div class="p-5 rounded-xl bg-amber-50/60 border border-amber-200 space-y-2">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-amber-900 flex items-center gap-2">
                                <i class="fa-solid fa-triangle-exclamation text-amber-600"></i> Absent Standard Protections
                            </h5>
                            <ul class="space-y-1.5 text-xs text-amber-800">
                                ${data.missing_clauses.map(c => `<li class="flex items-start gap-2"><span>⚠️</span><span>${escapeHtml(c)}</span></li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                `;
                showNotification('Risk audit completed', 'success');
            } catch (err) {
                console.error(err);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-sm">Risk audit failed: ${escapeHtml(err.message)}</div>`;
                showNotification(`Failed: ${err.message}`, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        // --- Workflow 3: Data & Metric Extraction ---
        async function runDataExtract() {
            if (documentsLoaded.length === 0) {
                showNotification('Please upload a PDF document first', 'warning');
                return;
            }

            const btn = document.getElementById('runExtractBtn');
            const copyBtn = document.getElementById('copyDataCsvBtn');
            const container = document.getElementById('extractContent');
            if (btn) btn.disabled = true;

            container.innerHTML = `
                <div class="p-8 text-center space-y-3">
                    <i class="fa-solid fa-circle-notch fa-spin text-4xl text-blue-600"></i>
                    <p class="text-sm font-semibold text-slate-700">Deconstructing document into quantitative metrics, dates, and entities...</p>
                </div>
            `;

            try {
                const resp = await fetch(`${API_BASE}/workflow/extract-data`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-User-Id': getUserId() },
                    body: JSON.stringify({})
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || 'Extraction failed');
                }

                const data = await resp.json();
                latestWorkflowData = { type: 'extract', data };
                currentExtractedPoints = data.data_points || [];
                lastReportMarkdown = compileDataReport(data);

                if (copyBtn) copyBtn.classList.remove('hidden');

                const categoryBadges = {
                    Financial: 'bg-emerald-50 text-emerald-700 border-emerald-200',
                    Metric: 'bg-blue-50 text-blue-700 border-blue-200',
                    Date: 'bg-purple-50 text-purple-700 border-purple-200',
                    Obligation: 'bg-amber-50 text-amber-700 border-amber-200',
                    Entity: 'bg-indigo-50 text-indigo-700 border-indigo-200'
                };

                container.innerHTML = `
                    <div class="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
                        <div class="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                            <span class="text-xs font-bold text-slate-700">${currentExtractedPoints.length} Structured Metrics Identified</span>
                            <span class="text-[10px] text-slate-500">Document: ${escapeHtml(data.document_name)}</span>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-xs">
                                <thead class="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
                                    <tr>
                                        <th class="py-2.5 px-4">Category</th>
                                        <th class="py-2.5 px-4">Metric Descriptor</th>
                                        <th class="py-2.5 px-4 font-bold text-slate-800">Value</th>
                                        <th class="py-2.5 px-4">Context & Significance</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-100 text-slate-700">
                                    ${currentExtractedPoints.map(dp => `
                                        <tr class="hover:bg-slate-50/80 transition">
                                            <td class="py-3 px-4">
                                                <span class="px-2 py-0.5 rounded-full border text-[10px] font-bold ${categoryBadges[dp.category] || 'bg-slate-100 text-slate-600'}">
                                                    ${escapeHtml(dp.category)}
                                                </span>
                                            </td>
                                            <td class="py-3 px-4 font-semibold text-slate-900">${escapeHtml(dp.label)}</td>
                                            <td class="py-3 px-4 font-mono font-bold text-indigo-600">${escapeHtml(dp.value)}</td>
                                            <td class="py-3 px-4 text-slate-600">${escapeHtml(dp.context)}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
                showNotification(`Extracted ${currentExtractedPoints.length} structured metrics`, 'success');
            } catch (err) {
                console.error(err);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-sm">Extraction failed: ${escapeHtml(err.message)}</div>`;
                showNotification(`Failed: ${err.message}`, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        function copyExtractedCsv() {
            if (!currentExtractedPoints || currentExtractedPoints.length === 0) return;
            const headers = ['Category', 'Descriptor', 'Value', 'Context', 'Source'];
            const rows = currentExtractedPoints.map(dp => [
                `"${dp.category}"`,
                `"${dp.label.replace(/"/g, '""')}"`,
                `"${dp.value.replace(/"/g, '""')}"`,
                `"${dp.context.replace(/"/g, '""')}"`,
                `"${dp.source_file.replace(/"/g, '""')}"`
            ].join(','));
            const csv = [headers.join(','), ...rows].join('\n');
            navigator.clipboard.writeText(csv).then(() => {
                showNotification('Table copied as CSV to clipboard!', 'success');
            });
        }

        // --- Workflow 4: Cross-Document Comparison ---
        async function runCompareDocuments() {
            if (documentsLoaded.length < 2) {
                showNotification('Please upload at least 2 PDF documents to run comparison', 'warning');
                return;
            }

            const doc1 = document.getElementById('compareDoc1').value;
            const doc2 = document.getElementById('compareDoc2').value;

            if (!doc1 || !doc2 || doc1 === doc2) {
                showNotification('Please select two distinct documents to compare', 'warning');
                return;
            }

            const btn = document.getElementById('runCompareBtn');
            const container = document.getElementById('compareContent');
            if (btn) btn.disabled = true;

            container.innerHTML = `
                <div class="p-8 text-center space-y-3">
                    <i class="fa-solid fa-circle-notch fa-spin text-4xl text-purple-600"></i>
                    <p class="text-sm font-semibold text-slate-700">Performing side-by-side comparative analysis...</p>
                    <p class="text-xs text-slate-500">Checking for contradictions, policy revisions, and divergent terms.</p>
                </div>
            `;

            try {
                const resp = await fetch(`${API_BASE}/workflow/compare`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-User-Id': getUserId() },
                    body: JSON.stringify({ file_id_1: doc1, file_id_2: doc2 })
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || 'Comparison failed');
                }

                const data = await resp.json();
                latestWorkflowData = { type: 'compare', data };
                lastReportMarkdown = compileCompareReport(data);

                const deltaBadge = {
                    Contradiction: 'bg-red-50 text-red-700 border-red-200',
                    Modified: 'bg-amber-50 text-amber-700 border-amber-200',
                    Addition: 'bg-blue-50 text-blue-700 border-blue-200',
                    Alignment: 'bg-emerald-50 text-emerald-700 border-emerald-200'
                };

                container.innerHTML = `
                    <div class="bg-gradient-to-r from-purple-50 to-indigo-50 p-5 rounded-2xl border border-purple-100 space-y-2">
                        <span class="text-xs uppercase font-bold text-purple-700 tracking-wider">Comparative Executive Summary</span>
                        <p class="text-sm text-slate-800 leading-relaxed">${escapeHtml(data.summary_matrix)}</p>
                    </div>

                    <div class="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
                        <div class="px-5 py-3 bg-slate-50 border-b border-slate-200 font-bold text-xs text-slate-700">
                            Side-by-Side Delta Matrix (${data.deltas.length} Points of Variation)
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-xs">
                                <thead class="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
                                    <tr>
                                        <th class="py-2.5 px-4">Topic / Dimension</th>
                                        <th class="py-2.5 px-4 font-bold text-slate-800">${escapeHtml(data.doc1_name)}</th>
                                        <th class="py-2.5 px-4 font-bold text-indigo-700">${escapeHtml(data.doc2_name)}</th>
                                        <th class="py-2.5 px-4">Delta Type</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-100 text-slate-700">
                                    ${data.deltas.map(d => `
                                        <tr class="hover:bg-slate-50/80 transition">
                                            <td class="py-3 px-4 font-bold text-slate-900">${escapeHtml(d.dimension)}</td>
                                            <td class="py-3 px-4 text-slate-600 bg-slate-50/40">${escapeHtml(d.doc1_content)}</td>
                                            <td class="py-3 px-4 text-slate-900 bg-purple-50/20 font-medium">${escapeHtml(d.doc2_content)}</td>
                                            <td class="py-3 px-4">
                                                <span class="px-2 py-0.5 rounded-full border text-[10px] font-bold ${deltaBadge[d.delta_type] || 'bg-slate-100 text-slate-600'}">
                                                    ${escapeHtml(d.delta_type)}
                                                </span>
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    ${(data.consensus_points && data.consensus_points.length > 0) ? `
                        <div class="p-4 bg-emerald-50/70 rounded-xl border border-emerald-200 space-y-1.5">
                            <h5 class="text-xs font-bold uppercase tracking-wider text-emerald-900 flex items-center gap-2">
                                <i class="fa-solid fa-circle-check text-emerald-600"></i> Points of Full Consensus
                            </h5>
                            <ul class="space-y-1 text-xs text-emerald-800">
                                ${data.consensus_points.map(cp => `<li class="flex items-start gap-2"><span>✓</span><span>${escapeHtml(cp)}</span></li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                `;
                showNotification('Cross-document comparison generated', 'success');
            } catch (err) {
                console.error(err);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-sm">Comparison failed: ${escapeHtml(err.message)}</div>`;
                showNotification(`Failed: ${err.message}`, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        // --- Workflow 5: Interactive Knowledge Quiz ---
        async function runQuiz() {
            if (documentsLoaded.length === 0) {
                showNotification('Please upload a PDF document first', 'warning');
                return;
            }

            const btn = document.getElementById('runQuizBtn');
            const scoreBadge = document.getElementById('quizScoreBadge');
            const container = document.getElementById('quizContent');
            if (btn) btn.disabled = true;

            container.innerHTML = `
                <div class="p-8 text-center space-y-3">
                    <i class="fa-solid fa-circle-notch fa-spin text-4xl text-emerald-600"></i>
                    <p class="text-sm font-semibold text-slate-700">Deconstructing document into interactive knowledge checks...</p>
                </div>
            `;

            try {
                const resp = await fetch(`${API_BASE}/workflow/quiz`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-User-Id': getUserId() },
                    body: JSON.stringify({ count: 5 })
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || 'Quiz generation failed');
                }

                const data = await resp.json();
                activeQuizQuestions = data.questions || [];
                currentQuizScore = 0;
                answeredQuestions.clear();

                if (scoreBadge) {
                    scoreBadge.classList.remove('hidden');
                    scoreBadge.textContent = `Score: 0 / ${activeQuizQuestions.length}`;
                }

                container.innerHTML = `
                    <div class="space-y-4 max-w-2xl mx-auto">
                        <div class="p-4 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between">
                            <span class="text-xs font-bold text-slate-800">${escapeHtml(data.topic)}</span>
                            <span class="text-xs text-slate-500">${activeQuizQuestions.length} Questions</span>
                        </div>
                        ${activeQuizQuestions.map((q, qIdx) => `
                            <div class="quiz-card p-5 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-3" id="qcard-${qIdx}">
                                <div class="flex items-start gap-3">
                                    <span class="w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">${qIdx + 1}</span>
                                    <h5 class="text-sm font-bold text-slate-900">${escapeHtml(q.question)}</h5>
                                </div>
                                <div class="space-y-2 pt-1">
                                    ${q.options.map((opt, optIdx) => `
                                        <button type="button" class="quiz-opt-btn w-full text-left p-3 rounded-lg border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/50 text-xs text-slate-700 transition flex items-center gap-3" data-qidx="${qIdx}" data-oidx="${optIdx}">
                                            <span class="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-bold text-slate-500">${String.fromCharCode(65 + optIdx)}</span>
                                            <span class="flex-1">${escapeHtml(opt)}</span>
                                        </button>
                                    `).join('')}
                                </div>
                                <div id="qfeedback-${qIdx}" class="hidden p-3 rounded-lg text-xs space-y-1"></div>
                            </div>
                        `).join('')}
                    </div>
                `;

                // Attach option click handlers
                container.querySelectorAll('.quiz-opt-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const qIdx = parseInt(btn.dataset.qidx);
                        const oIdx = parseInt(btn.dataset.oidx);
                        handleQuizOptionSelect(qIdx, oIdx);
                    });
                });

                showNotification('Interactive quiz generated', 'success');
            } catch (err) {
                console.error(err);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-sm">Quiz generation failed: ${escapeHtml(err.message)}</div>`;
                showNotification(`Failed: ${err.message}`, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
        }

        function handleQuizOptionSelect(qIdx, oIdx) {
            if (answeredQuestions.has(qIdx)) return;
            answeredQuestions.add(qIdx);

            const q = activeQuizQuestions[qIdx];
            const card = document.getElementById(`qcard-${qIdx}`);
            const feedback = document.getElementById(`qfeedback-${qIdx}`);
            const scoreBadge = document.getElementById('quizScoreBadge');

            const isCorrect = oIdx === q.correct_index;
            if (isCorrect) currentQuizScore++;

            if (scoreBadge) {
                scoreBadge.textContent = `Score: ${currentQuizScore} / ${activeQuizQuestions.length}`;
            }

            // Style all buttons for this question
            card.querySelectorAll('.quiz-opt-btn').forEach(btn => {
                const optIndex = parseInt(btn.dataset.oidx);
                btn.disabled = true;
                if (optIndex === q.correct_index) {
                    btn.className = 'quiz-opt-btn w-full text-left p-3 rounded-lg border-2 border-emerald-500 bg-emerald-50 text-xs font-semibold text-emerald-900 flex items-center gap-3';
                } else if (optIndex === oIdx && !isCorrect) {
                    btn.className = 'quiz-opt-btn w-full text-left p-3 rounded-lg border-2 border-red-500 bg-red-50 text-xs font-semibold text-red-900 flex items-center gap-3';
                } else {
                    btn.className = 'quiz-opt-btn w-full text-left p-3 rounded-lg border border-slate-200 text-xs text-slate-400 opacity-60 flex items-center gap-3';
                }
            });

            feedback.classList.remove('hidden');
            if (isCorrect) {
                feedback.className = 'p-3 rounded-lg text-xs space-y-1 bg-emerald-50 border border-emerald-200 text-emerald-800';
                feedback.innerHTML = `<p class="font-bold">✓ Correct!</p><p>${escapeHtml(q.explanation)}</p><p class="text-[10px] text-emerald-600 font-mono">Source: ${escapeHtml(q.citation)}</p>`;
            } else {
                feedback.className = 'p-3 rounded-lg text-xs space-y-1 bg-red-50 border border-red-200 text-red-800';
                feedback.innerHTML = `<p class="font-bold">✗ Incorrect</p><p>${escapeHtml(q.explanation)}</p><p class="text-[10px] text-red-600 font-mono">Source: ${escapeHtml(q.citation)}</p>`;
            }
        }

        // --- Evidence Inspector Modal Handlers ---
        let currentEvidenceText = '';

        function openEvidenceModal(sourceFile, snippet) {
            currentEvidenceText = snippet || '';
            const modal = document.getElementById('evidenceModal');
            const fileEl = document.getElementById('evidenceSourceFile');
            const snippetEl = document.getElementById('evidenceSnippet');

            if (fileEl) fileEl.textContent = sourceFile || 'Source Excerpt';
            if (snippetEl) snippetEl.textContent = snippet || 'Excerpt content unavailable.';
            if (modal) modal.classList.remove('hidden');
        }

        function closeEvidenceModal() {
            const modal = document.getElementById('evidenceModal');
            if (modal) modal.classList.add('hidden');
        }

        function copyEvidenceText() {
            if (!currentEvidenceText) return;
            navigator.clipboard.writeText(currentEvidenceText).then(() => {
                showNotification('Source citation copied to clipboard', 'success');
            });
        }

        // --- Export Executive Report Modal Handlers ---
        function openExportModal() {
            const modal = document.getElementById('exportModal');
            const textarea = document.getElementById('exportReportText');
            
            // Build report markdown from current active tab
            let report = '';
            if (currentActiveTab === 'summary' && lastReportMarkdown) {
                report = lastReportMarkdown;
            } else if (currentActiveTab === 'risks' && lastReportMarkdown) {
                report = lastReportMarkdown;
            } else if (currentActiveTab === 'extract' && lastReportMarkdown) {
                report = lastReportMarkdown;
            } else if (currentActiveTab === 'compare' && lastReportMarkdown) {
                report = lastReportMarkdown;
            } else {
                report = compileChatReport();
            }

            if (textarea) textarea.value = report;
            if (modal) modal.classList.remove('hidden');
        }

        function closeExportModal() {
            const modal = document.getElementById('exportModal');
            if (modal) modal.classList.add('hidden');
        }

        function copyReportText() {
            const textarea = document.getElementById('exportReportText');
            if (!textarea || !textarea.value) return;
            navigator.clipboard.writeText(textarea.value).then(() => {
                showNotification('Executive Briefing copied to clipboard!', 'success');
                closeExportModal();
            });
        }

        // --- Report Compilers ---
        function compileSummaryReport(data) {
            return `# ${data.title}\n\n` +
                `**Generated by DocuMind Studio** | Date: ${new Date().toLocaleDateString()}\n\n` +
                `## Executive Summary\n${data.executive_summary}\n\n` +
                `## Key Findings\n${(data.key_findings || []).map(f => `- ${f}`).join('\n')}\n\n` +
                `## Critical Decisions\n${(data.critical_decisions || []).map(d => `- ${d}`).join('\n')}\n\n` +
                `## Timeline & Milestones\n${(data.timeline_or_milestones || []).map(m => `- ${m}`).join('\n')}\n\n` +
                `## Action Items\n${(data.action_items || []).map(a => `- [ ] ${a}`).join('\n')}\n\n` +
                `---\n*Sources: ${(data.sources || []).join(', ')}*`;
        }

        function compileRiskReport(data) {
            return `# DocuMind Risk & Compliance Audit Matrix\n\n` +
                `**Overall Risk Posture:** ${data.overall_risk_level}\n` +
                `*${data.risk_score_narrative}*\n\n` +
                `## Identified Risks (${data.risks.length})\n\n` +
                data.risks.map(r => 
                    `### [${r.severity} Severity] ${r.title} (${r.category})\n` +
                    `> "${r.clause_excerpt}"\n\n` +
                    `- **Impact:** ${r.impact}\n` +
                    `- **Mitigation:** ${r.mitigation}\n`
                ).join('\n') +
                `\n## Absent Standard Protections\n` +
                (data.missing_clauses || []).map(c => `- ${c}`).join('\n');
        }

        function compileDataReport(data) {
            let md = `# DocuMind Quantitative Data & Metrics Report\n\nDocument: ${data.document_name}\n\n`;
            md += `| Category | Metric Descriptor | Value | Context |\n| :--- | :--- | :--- | :--- |\n`;
            (data.data_points || []).forEach(dp => {
                md += `| ${dp.category} | ${dp.label} | ${dp.value} | ${dp.context} |\n`;
            });
            return md;
        }

        function compileCompareReport(data) {
            let md = `# DocuMind Cross-Document Comparative Report\n\n` +
                `Comparing: **${data.doc1_name}** vs. **${data.doc2_name}**\n\n` +
                `## Executive Delta Summary\n${data.summary_matrix}\n\n` +
                `## Comparison Matrix\n\n` +
                `| Dimension | ${data.doc1_name} | ${data.doc2_name} | Delta Type |\n| :--- | :--- | :--- | :--- |\n`;
            data.deltas.forEach(d => {
                md += `| ${d.dimension} | ${d.doc1_content} | ${d.doc2_content} | ${d.delta_type} |\n`;
            });
            return md;
        }

        function compileChatReport() {
            let md = `# DocuMind Grounded Q&A Briefing\n\nGenerated: ${new Date().toLocaleString()}\n\n`;
            conversationHistory.forEach((msg, i) => {
                if (msg.role === 'user') {
                    md += `### Q: ${msg.query}\n\n`;
                } else if (msg.role === 'assistant') {
                    md += `${msg.answer}\n\n`;
                    if (msg.sources && msg.sources.length > 0) {
                        md += `*Sources: ${msg.sources.map(s => s.source).join(', ')}*\n\n`;
                    }
                }
            });
            return md;
        }

        document.addEventListener('DOMContentLoaded', async () => {
            await initAnnaRuntime();
            setupEventListeners();
            startNewConversation();
            loadDocuments();
            loadStats();
            setInterval(loadStats, 5000);
        });
