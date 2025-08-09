# services/web/Dockerfile
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy HTML files
COPY index.html /usr/share/nginx/html/
COPY models.html /usr/share/nginx/html/
COPY document.html /usr/share/nginx/html/
# Copy static assets
COPY static /usr/share/nginx/html/static/

# Create a health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Expose port 80 (nginx default)
EXPOSE 80

# nginx will start automatically from the base image
