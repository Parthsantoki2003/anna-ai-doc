        const API_BASE = 'https://anna-document-workspace-wj7amu7tcq-uc.a.run.app';
        let currentConversationId = null;
        let conversationHistory = [];
        let documentsLoaded = [];
        let uploadInProgress = false;
        let annaRuntime = null;

        async function initAnnaRuntime() {
            try {
                if (typeof window !== 'undefined' && window.AnnaAppRuntime) {
                    const result = await Promise.resolve(window.AnnaAppRuntime.connect?.());
                    annaRuntime = result;
                }
            } catch (e) {
                console.warn('Anna Runtime not available:', e);
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
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                document.getElementById('fileNameDisplay').textContent = `Selected ${file.name}`;
                uploadFile(file);
            }
        }

        function uploadSelectedFile() {
            const input = document.getElementById('fileInput');
            const file = input.files && input.files[0];
            if (file) {
                uploadFile(file);
            } else {
                document.getElementById('fileInput').click();
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
            if (files.length > 0) uploadFile(files[0]);
        }

        async function annaHostUpload(file) {
            try {
                if (!annaRuntime) throw new Error('Anna Runtime not available');
                
                if (file.size <= 8 * 1024 * 1024) {
                    const bytes = new Uint8Array(await file.arrayBuffer());
                    let binary = '';
                    const step = 0x8000;
                    for (let i = 0; i < bytes.length; i += step) {
                        binary += String.fromCharCode(...bytes.subarray(i, i + step));
                    }
                    const response = await annaRuntime.upload?.inline?.({
                        bytes_b64: btoa(binary),
                        mime: 'application/pdf',
                        purpose: 'document-ingestion',
                        filename: file.name
                    });
                    if (!response?.ok) throw new Error(response?.error?.message || 'Upload failed');
                    return response.result.url;
                }

                const negotiated = await annaRuntime.upload?.negotiate?.({
                    mime: 'application/pdf',
                    purpose: 'document-ingestion',
                    size_bytes: file.size
                });
                if (!negotiated?.ok) throw new Error('Upload negotiation failed');
                
                const uploadResponse = await fetch(negotiated.result.upload_url, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/pdf' },
                    body: file
                });
                if (!uploadResponse.ok) throw new Error('File upload failed');
                
                const confirmed = await annaRuntime.upload?.confirm?.({ file_ref: negotiated.result.file_ref });
                if (!confirmed?.ok) throw new Error('Upload confirmation failed');
                return confirmed.result.url;
            } catch (error) {
                throw new Error(`Anna upload failed: ${error.message}`);
            }
        }

        async function uploadFile(file) {
            if (uploadInProgress) return;
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showNotification('Only PDF files are supported', 'error');
                return;
            }
            uploadInProgress = true;

            document.getElementById('fileProgress').classList.remove('hidden');
            document.getElementById('fileNameDisplay').textContent = `Uploading ${file.name}...`;

            try {
                const hostedUrl = await annaHostUpload(file);
                const response = await fetch(`${API_BASE}/ingest-from-url`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_url: hostedUrl,
                        file_name: file.name,
                        chunk_strategy: 'recursive',
                        chunk_size: 500,
                        chunk_overlap: 50
                    })
                });

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
            }
        }

        async function loadDocuments() {
            try {
                const response = await fetch(`${API_BASE}/documents`);
                if (!response.ok) throw new Error(`Documents request failed (${response.status})`);
                const documents = await response.json();
                documentsLoaded = documents;
                renderDocuments(documents);
            } catch (error) {
                console.error('Failed to load documents:', error);
            }
        }

        function renderDocuments(documents) {
            const list = document.getElementById('documentsList');
            if (documents.length === 0) {
                list.innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No documents uploaded yet</p>';
                return;
            }

            list.innerHTML = documents.map(doc => `
                <div class="document-card bg-slate-50 rounded-lg p-3 flex items-start justify-between hover:bg-slate-100 transition">
                    <div class="flex-1">
                        <p class="text-sm font-semibold text-slate-700 truncate" title="${escapeHtml(doc.file_name)}"><i class="fa-solid fa-file-pdf text-red-500"></i> ${escapeHtml(doc.file_name)}</p>
                        <p class="text-xs text-slate-500 mt-1">${doc.chunks_count} indexed chunks</p>
                    </div>
                    <button type="button" class="text-red-500 hover:text-red-700 hover:bg-red-50 p-2 rounded transition" title="Delete document" data-delete="${escapeHtml(doc.file_id)}" data-name="${escapeHtml(doc.file_name)}">
                        <i class="fa-solid fa-trash-alt"></i>
                    </button>
                </div>
            `).join('');

            list.querySelectorAll('[data-delete]').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    deleteDocument(btn.dataset.delete, btn.dataset.name);
                });
            });
        }

        async function deleteDocument(fileId, fileName) {
            if (!confirm(`Delete "${fileName}"? This will remove it from the knowledge base.`)) return;

            try {
                const response = await fetch(`${API_BASE}/documents/${fileId}`, {
                    method: 'DELETE'
                });

                if (!response.ok) throw new Error('Delete failed');

                showNotification(`Document "${fileName}" deleted`, 'success');
                loadDocuments();
                loadStats();
            } catch (error) {
                showNotification(`Delete failed: ${error.message}`, 'error');
            }
        }

        async function loadStats() {
            try {
                const response = await fetch(`${API_BASE}/stats`);
                if (!response.ok) throw new Error(`Stats request failed (${response.status})`);
                const stats = await response.json();
                document.getElementById('statsChunks').textContent = stats.total_chunks;
                document.getElementById('statsDocuments').textContent = stats.total_documents;
            } catch (error) {
                console.error('Failed to load stats:', error);
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
            document.getElementById('conversationHistory').innerHTML = '<p class="text-sm text-slate-500 text-center py-4">No conversation yet</p>';
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
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        top_k: 5,
                        generate_answer: true,
                        conversation_context: conversationHistory
                    })
                });

                if (!response.ok) throw new Error('Query failed');

                const data = await response.json();
                
                document.getElementById(loadingId)?.remove();

                conversationHistory.push({
                    role: 'user',
                    query,
                    timestamp: new Date().toISOString()
                });

                conversationHistory.push({
                    role: 'assistant',
                    answer: data.answer,
                    sources: data.sources,
                    timestamp: new Date().toISOString()
                });

                currentConversationId = data.conversation_id;

                addAssistantMessage(data.answer, data.sources, data.processing_time);
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