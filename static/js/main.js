/* ==========================================================================
   Crop Health AI - Dynamic Interactions & 3D Parallax Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------------
    // 1. Uploader & Drag-and-Drop Handler
    // ----------------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-upload');
    const imagePreview = document.getElementById('image-preview');
    const previewStage = document.getElementById('preview-stage');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const actions = document.getElementById('actions');
    const btnCancel = document.getElementById('btn-cancel');
    const uploadForm = document.getElementById('upload-form');
    const loadingState = document.getElementById('loading-state');
    
    if (dropZone && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop zone on drag hover
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        // Handle file input selection
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                handleFile(this.files[0]);
            }
        });

        // Cancel selection
        if (btnCancel) {
            btnCancel.addEventListener('click', () => {
                resetUploader();
            });
        }

        // Form submission loading state
        if (uploadForm) {
            uploadForm.addEventListener('submit', () => {
                dropZone.classList.add('hidden');
                actions.classList.add('hidden');
                if (loadingState) {
                    loadingState.classList.remove('hidden');
                }
            });
        }
    }

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropZone.classList.add('dragover');
    }

    function unhighlight() {
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files && files[0]) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPEG or PNG).');
            return;
        }

        const reader = new FileReader();

        reader.onload = function(e) {
            if (imagePreview) {
                imagePreview.src = e.target.result;
            }
            if (previewStage) {
                previewStage.classList.remove('hidden');
            }
            if (dropZoneContent) {
                dropZoneContent.classList.add('hidden');
            }
            if (dropZone) {
                dropZone.style.padding = '12px';
                dropZone.style.borderStyle = 'solid';
                dropZone.style.borderColor = 'rgba(0, 245, 160, 0.5)';
            }
            if (actions) {
                actions.classList.remove('hidden');
            }
        };

        reader.readAsDataURL(file);
    }

    function resetUploader() {
        if (fileInput) fileInput.value = '';
        if (imagePreview) imagePreview.src = '';
        if (previewStage) previewStage.classList.add('hidden');
        if (dropZoneContent) dropZoneContent.classList.remove('hidden');
        if (dropZone) {
            dropZone.style.padding = '48px 24px';
            dropZone.style.borderStyle = 'dashed';
            dropZone.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        }
        if (actions) actions.classList.add('hidden');
    }

    // ----------------------------------------------------------------------
    // 2. Lightweight GPU-Accelerated 3D Tilt Effect
    // ----------------------------------------------------------------------
    const tiltCards = document.querySelectorAll('.glass-card, .main-scanner-card');
    
    tiltCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            // Maximum tilt angle in degrees
            const maxTilt = 6;

            const tiltX = ((y - centerY) / centerY) * -maxTilt;
            const tiltY = ((x - centerX) / centerX) * maxTilt;

            card.style.transform = `perspective(1000px) rotateX(${tiltX.toFixed(2)}deg) rotateY(${tiltY.toFixed(2)}deg) scale3d(1.01, 1.01, 1.01)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
        });
    });

    // ----------------------------------------------------------------------
    // 3. Ambient Particle Spawns
    // ----------------------------------------------------------------------
    const particlesContainer = document.getElementById('particles');
    if (particlesContainer) {
        const particleCount = 18;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle-dot';
            
            const size = Math.random() * 3 + 1;
            const posX = Math.random() * 100;
            const posY = Math.random() * 100;
            const duration = Math.random() * 15 + 10;
            const delay = Math.random() * 5;

            particle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background-color: rgba(0, 245, 160, ${Math.random() * 0.4 + 0.2});
                border-radius: 50%;
                top: ${posY}%;
                left: ${posX}%;
                box-shadow: 0 0 10px rgba(0, 245, 160, 0.8);
                animation: floatParticle ${duration}s linear infinite;
                animation-delay: -${delay}s;
            `;
            particlesContainer.appendChild(particle);
        }

        // Add particle floating keyframes dynamically
        const styleSheet = document.createElement('style');
        styleSheet.textContent = `
            @keyframes floatParticle {
                0% { transform: translateY(0) translateX(0); opacity: 0.2; }
                50% { transform: translateY(-40px) translateX(20px); opacity: 0.8; }
                100% { transform: translateY(-80px) translateX(-10px); opacity: 0; }
            }
        `;
        document.head.appendChild(styleSheet);
    }

    // ----------------------------------------------------------------------
    // 4. Result Page Confidence Progress Bar Animation
    // ----------------------------------------------------------------------
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        const targetWidth = progressBar.getAttribute('data-confidence') || progressBar.style.width;
        if (targetWidth) {
            progressBar.style.width = '0%';
            setTimeout(() => {
                progressBar.style.width = targetWidth;
            }, 150);
        }
    }
});
