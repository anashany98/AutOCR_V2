// ==========================================================================
// AutoOCR - Main JavaScript
// Obsidian Flow Design System Integration
// ==========================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initTooltips();
    initDropdowns();
    initUploadZone();
    initSearchWithDebounce();
    initPagination();
    initKeyboardShortcuts();
    initAutoResize();
    initConfirmDialogs();
    
    console.log('AutoOCR UI initialized');
});

// --------------------------------------------------------------------------
// Tooltips
// --------------------------------------------------------------------------
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// --------------------------------------------------------------------------
// Dropdowns
// --------------------------------------------------------------------------
function initDropdowns() {
    document.querySelectorAll('.dropdown-toggle').forEach(function(dropdown) {
        dropdown.addEventListener('click', function(e) {
            e.preventDefault();
            const dropdownEl = this.closest('.dropdown');
            dropdownEl.classList.toggle('open');
        });
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.dropdown')) {
            document.querySelectorAll('.dropdown.open').forEach(function(dropdown) {
                dropdown.classList.remove('open');
            });
        }
    });
}

// --------------------------------------------------------------------------
// Upload Zone with Drag & Drop
// --------------------------------------------------------------------------
function initUploadZone() {
    const uploadZone = document.querySelector('.upload-zone');
    const fileInput = document.getElementById('file_input');
    
    if (!uploadZone || !fileInput) return;

    // Click to upload
    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function(eventName) {
        uploadZone.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(function(eventName) {
        uploadZone.addEventListener(eventName, function() {
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(function(eventName) {
        uploadZone.addEventListener(eventName, function() {
            uploadZone.classList.remove('dragover');
        });
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', function(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileCount(files.length);
            showFileList(files);
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            updateFileCount(this.files.length);
            showFileList(this.files);
        }
    });

    function updateFileCount(count) {
        const fileCountEl = document.getElementById('file-count');
        if (fileCountEl) {
            fileCountEl.textContent = `${count} archivo(s) seleccionado(s)`;
            fileCountEl.style.display = 'block';
        }
    }

    function showFileList(files) {
        const container = document.getElementById('file-list');
        if (!container) return;
        
        container.innerHTML = '';
        
        Array.from(files).forEach(function(file) {
            const item = document.createElement('div');
            item.className = 'file-item';
            item.innerHTML = `
                <i class="fas fa-file-${getFileIcon(file.name)}"></i>
                <span>${file.name}</span>
                <small>${formatFileSize(file.size)}</small>
            `;
            container.appendChild(item);
        });
    }

    function getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            pdf: 'pdf text-danger',
            doc: 'word text-primary',
            docx: 'word text-primary',
            xls: 'excel text-success',
            xlsx: 'excel text-success',
            jpg: 'image text-info',
            jpeg: 'image text-info',
            png: 'image text-info',
            gif: 'image text-info'
        };
        return icons[ext] || 'alt';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// --------------------------------------------------------------------------
// Search with Debounce (AJAX)
// --------------------------------------------------------------------------
function initSearchWithDebounce() {
    const searchInput = document.getElementById('search');
    if (!searchInput) return;

    let searchTimeout;
    const minLength = 2;
    const debounceDelay = 400;

    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        
        const query = this.value.trim();
        
        // Clear previous results if query is empty
        if (query.length === 0) {
            const form = this.closest('form');
            if (form) {
                form.submit();
            }
            return;
        }

        // Minimum characters before search
        if (query.length < minLength) {
            return;
        }

        searchTimeout = setTimeout(function() {
            // Check if we should use AJAX or form submit
            const useAjax = document.body.dataset.ajaxSearch === 'true';
            
            if (useAjax) {
                performAjaxSearch(query);
            } else {
                const form = searchInput.closest('form');
                if (form) {
                    form.submit();
                }
            }
        }, debounceDelay);
    });

    function performAjaxSearch(query) {
        const searchResults = document.getElementById('search-results');
        const loadingIndicator = document.getElementById('search-loading');
        
        if (!searchResults) return;
        
        if (loadingIndicator) loadingIndicator.style.display = 'block';
        
        const baseUrl = window.location.pathname;
        const params = new URLSearchParams(window.location.search);
        params.set('search', query);
        
        fetch(`${baseUrl}?${params.toString()}`, {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(function(response) {
            return response.text();
        })
        .then(function(html) {
            // Parse the response and update the results
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newResults = doc.getElementById('documents-table');
            
            if (newResults) {
                const currentResults = document.getElementById('documents-table');
                if (currentResults) {
                    currentResults.innerHTML = newResults.innerHTML;
                }
            }
            
            // Update URL without reload
            window.history.replaceState({}, '', `${baseUrl}?${params.toString()}`);
        })
        .catch(function(error) {
            console.error('Search error:', error);
            showToast('Error al realizar la búsqueda', 'error');
        })
        .finally(function() {
            if (loadingIndicator) loadingIndicator.style.display = 'none';
        });
    }
}

// --------------------------------------------------------------------------
// Pagination
// --------------------------------------------------------------------------
function initPagination() {
    document.querySelectorAll('.pagination').forEach(function(pagination) {
        pagination.addEventListener('click', function(e) {
            const pageLink = e.target.closest('.pagination-item');
            if (!pageLink || pageLink.classList.contains('active') || pageLink.disabled) {
                return;
            }

            const page = pageLink.dataset.page;
            if (!page) return;

            // Check for AJAX pagination
            if (document.body.dataset.ajaxPagination === 'true') {
                loadPageAjax(page);
            } else {
                // Traditional pagination - update URL and reload
                const params = new URLSearchParams(window.location.search);
                params.set('page', page);
                window.location.href = `${window.location.pathname}?${params.toString()}`;
            }
        });
    });

    function loadPageAjax(page) {
        const tableContainer = document.getElementById('documents-container');
        if (!tableContainer) return;

        // Show loading skeleton
        tableContainer.innerHTML = generateSkeletonTable();

        const params = new URLSearchParams(window.location.search);
        params.set('page', page);

        fetch(`${window.location.pathname}?${params.toString()}`, {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(function(response) {
            return response.text();
        })
        .then(function(html) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newContent = doc.getElementById('documents-container');
            
            if (newContent) {
                tableContainer.innerHTML = newContent.innerHTML;
            }
            
            // Update URL
            window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
            
            // Scroll to top of table
            tableContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
        .catch(function(error) {
            console.error('Pagination error:', error);
            showToast('Error al cargar la página', 'error');
        });
    }
}

function generateSkeletonTable() {
    return `
        <div class="table-container">
            <table class="table">
                <thead>
                    <tr>
                        <th><div class="skeleton skeleton-text" style="width: 20px;"></div></th>
                        <th><div class="skeleton skeleton-text" style="width: 150px;"></div></th>
                        <th><div class="skeleton skeleton-text" style="width: 100px;"></div></th>
                        <th><div class="skeleton skeleton-text" style="width: 80px;"></div></th>
                        <th><div class="skeleton skeleton-text" style="width: 120px;"></div></th>
                    </tr>
                </thead>
                <tbody>
                    ${Array(10).fill(`
                        <tr>
                            <td><div class="skeleton skeleton-text" style="width: 20px;"></div></td>
                            <td><div class="skeleton skeleton-text"></div></td>
                            <td><div class="skeleton skeleton-text" style="width: 80px;"></div></td>
                            <td><div class="skeleton skeleton-text" style="width: 60px;"></div></td>
                            <td><div class="skeleton skeleton-text" style="width: 100px;"></div></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// --------------------------------------------------------------------------
// Keyboard Shortcuts
// --------------------------------------------------------------------------
function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ignore if typing in input/textarea
        const tag = e.target.tagName.toLowerCase();
        const isEditing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
        
        // Ctrl/Cmd + K: Global search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const globalSearch = document.getElementById('globalSearch');
            if (globalSearch) {
                globalSearch.focus();
                globalSearch.select();
            } else {
                const searchInput = document.getElementById('search');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
        }
        
        // Ctrl/Cmd + N: New document
        if ((e.ctrlKey || e.metaKey) && e.key === 'n' && !isEditing) {
            e.preventDefault();
            const uploadLink = document.querySelector('a[href*="upload"]');
            if (uploadLink) {
                window.location.href = uploadLink.href;
            }
        }
        
        // Ctrl/Cmd + /: Show shortcuts help
        if ((e.ctrlKey || e.metaKey) && e.key === '/' && !isEditing) {
            e.preventDefault();
            showKeyboardShortcutsHelp();
        }
        
        // Escape: Close modals, clear search
        if (e.key === 'Escape') {
            // Close dropdowns
            document.querySelectorAll('.dropdown.open').forEach(function(dropdown) {
                dropdown.classList.remove('open');
            });
            
            // Close modals
            document.querySelectorAll('.modal-overlay.open').forEach(function(modal) {
                modal.classList.remove('open');
            });
            
            // Clear search if focused
            if (document.activeElement && document.activeElement.id === 'search') {
                document.activeElement.value = '';
            }
        }
        
        // Ctrl/Cmd + Enter: Submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && isEditing) {
            const form = e.target.closest('form');
            if (form) {
                form.submit();
            }
        }
    });
    
    function showKeyboardShortcutsHelp() {
        const shortcuts = [
            { key: 'Ctrl + K', action: 'Abrir búsqueda' },
            { key: 'Ctrl + N', action: 'Nuevo documento' },
            { key: 'Ctrl + /', action: 'Mostrar atajos' },
            { key: 'Escape', action: 'Cerrar/cancelar' },
            { key: 'Ctrl + Enter', action: 'Enviar formulario' }
        ];
        
        let html = '<div class="keyboard-shortcuts-list">';
        shortcuts.forEach(function(s) {
            html += `
                <div class="shortcut-item">
                    <kbd>${s.key}</kbd>
                    <span>${s.action}</span>
                </div>
            `;
        });
        html += '</div>';
        
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'modal-overlay open';
        modal.innerHTML = `
            <div class="modal" style="max-width: 400px;">
                <div class="modal-header">
                    <h5 class="modal-title">Atajos de Teclado</h5>
                    <button class="btn-close" onclick="this.closest('.modal-overlay').remove()"></button>
                </div>
                <div class="modal-body">
                    ${html}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
}

// --------------------------------------------------------------------------
// Auto-resize Textarea
// --------------------------------------------------------------------------
function initAutoResize() {
    const autoResize = function(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    };
    
    document.querySelectorAll('textarea[data-auto-resize]').forEach(function(textarea) {
        textarea.addEventListener('input', function() {
            autoResize(this);
        });
        
        // Initial resize
        autoResize(textarea);
    });
}

// --------------------------------------------------------------------------
// Confirmation Dialogs
// --------------------------------------------------------------------------
function initConfirmDialogs() {
    document.querySelectorAll('[data-confirm]').forEach(function(element) {
        element.addEventListener('click', function(e) {
            const message = this.dataset.confirm || '¿Estás seguro?';
            if (!confirm(message)) {
                e.preventDefault();
                return false;
            }
        });
    });
    
    document.querySelectorAll('form[data-confirm]').forEach(function(form) {
        form.addEventListener('submit', function(e) {
            const message = form.dataset.confirm;
            if (message && !confirm(message)) {
                e.preventDefault();
                return false;
            }
        });
    });
}

// --------------------------------------------------------------------------
// Utility Functions
// --------------------------------------------------------------------------

// Show toast notification
window.showToast = function(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas fa-${icons[type] || 'info-circle'}"></i>
        <span>${message}</span>
        <button class="btn-close ms-auto" onclick="this.parentElement.remove()"></button>
    `;
    
    container.appendChild(toast);
    
    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(function() {
            toast.remove();
        }, 300);
    }, duration);
};

// Copy to clipboard
window.copyToClipboard = function(text, message = 'Copiado al portapapeles') {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function() {
            showToast(message, 'success');
        }).catch(function(err) {
            console.error('Error copying:', err);
            showToast('Error al copiar', 'error');
        });
    } else {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast(message, 'success');
    }
};

// Format file size
window.formatFileSize = function(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Loading state for buttons
window.setButtonLoading = function(button, loading) {
    if (!button) return;
    
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = '<span class="spinner"></span> Procesando...';
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalText || button.innerHTML;
    }
};

// Export functions for global use
window.AutoOCR = {
    showToast: window.showToast,
    copyToClipboard: window.copyToClipboard,
    formatFileSize: window.formatFileSize,
    setButtonLoading: window.setButtonLoading
};
