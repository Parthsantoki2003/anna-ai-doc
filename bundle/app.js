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

            if (documents.length === 0) {
                list.innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No documents uploaded yet</p>';
                return;
            }

            list.innerHTML = documents.map(doc => `
                <div class="document-card bg-slate-50 rounded-lg p-3 flex items-start justify-between hover:bg-slate-100 transition border border-slate-200">
                    <div class="flex-1 min-w-0 pr-2">
                        <p class="text-sm font-semibold text-slate-700 truncate" title="${escapeHtml(doc.file_name)}">
                            <i class="fa-solid fa-file-pdf text-red-500 mr-1.5"></i>${escapeHtml(doc.file_name)}
                        </p>
                        <p class="text-xs text-slate-500 mt-1">${doc.chunks_count} indexed chunks</p>
                    </div>
                    <button type="button" class="delete-doc-btn flex-shrink-0 text-red-500 hover:text-red-700 hover:bg-red-50 p-2 rounded transition" title="Delete document" data-delete="${escapeHtml(doc.file_id)}" data-name="${escapeHtml(doc.file_name)}">
                        <i class="fa-solid fa-trash-alt pointer-events-none"></i>
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

        function addAssistantMessage(answer, sources, processingTime) {
            const container = document.getElementById('messagesContainer');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message-enter flex justify-start';
            
            messageDiv.innerHTML = `
                <div class="chat-bubble assistant">
                    <div class="chat-meta"><span class="avatar">A</span>Anna</div>
                    <div class="chat-content leading-relaxed mb-2">${renderMarkdown(answer || 'No answer generated')}</div>
                    <p class="text-xs text-slate-500 mt-3 pt-3 border-t border-slate-200">Response time: ${processingTime.toFixed(2)}s</p>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
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

        document.addEventListener('DOMContentLoaded', async () => {
            await initAnnaRuntime();
            setupEventListeners();
            startNewConversation();
            loadDocuments();
            loadStats();
            setInterval(loadStats, 5000);
        });
