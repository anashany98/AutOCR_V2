// AutOCR Web Interface JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // File upload validation
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const files = e.target.files;
            let totalSize = 0;
            const maxSize = 50 * 1024 * 1024; // 50MB
            const allowedTypes = ['.pdf', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'];

            for (let file of files) {
                totalSize += file.size;

                const extension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
                if (!allowedTypes.includes(extension)) {
                    alert(`Tipo de archivo no permitido: ${file.name}`);
                    e.target.value = '';
                    return;
                }
            }

            if (totalSize > maxSize) {
                alert('El tamaño total de los archivos excede el límite de 50MB');
                e.target.value = '';
                return;
            }

            // Show file count
            const fileCount = document.createElement('small');
            fileCount.className = 'text-muted';
            fileCount.id = 'file-count';
            fileCount.textContent = `${files.length} archivo(s) seleccionado(s)`;

            const existing = document.getElementById('file-count');
            if (existing) {
                existing.remove();
            }

            fileInput.parentNode.appendChild(fileCount);
        });
    }

    // Search functionality
    const searchInput = document.getElementById('search');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(function() {
                // Auto-submit search after 500ms of no typing
                const form = searchInput.closest('form');
                if (form) {
                    form.submit();
                }
            }, 500);
        });
    }

    // Confirm batch processing
    const batchProcessBtn = document.querySelector('a[href*="batch_process"]');
    if (batchProcessBtn) {
        batchProcessBtn.addEventListener('click', function(e) {
            if (!confirm('¿Está seguro de que desea ejecutar el procesamiento batch? Esto puede tardar varios minutos.')) {
                e.preventDefault();
            }
        });
    }

    // Loading states for forms
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Procesando...';
            }
        });
    });

    // Table row highlighting
    const tableRows = document.querySelectorAll('tbody tr');
    tableRows.forEach(function(row) {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f8f9fa';
        });
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });

    // Copy to clipboard functionality
    window.copyToClipboard = function() {
        const textElement = document.querySelector('pre');
        if (textElement && navigator.clipboard) {
            navigator.clipboard.writeText(textElement.textContent).then(function() {
                showNotification('Texto copiado al portapapeles', 'success');
            }).catch(function(err) {
                console.error('Error copying text: ', err);
                showNotification('Error al copiar el texto', 'error');
            });
        }
    };

    // Notification system
    function showNotification(message, type) {
        const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
        const icon = type === 'success' ? 'fa-check' : 'fa-exclamation-triangle';

        const alert = document.createElement('div');
        alert.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alert.innerHTML = `
            <i class="fas ${icon}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alert);

        setTimeout(function() {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 3000);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search focus
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('search');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }

        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.getElementById('search');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.value = '';
                searchInput.closest('form').submit();
            }
        }
    });

    console.log('AutOCR Web Interface loaded successfully');
});